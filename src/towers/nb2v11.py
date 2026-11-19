# %% [markdown]
# # FLAG 2027 — Notebook 2 v8 = v7 + face/voice encoders fine-tuned cross-modally on VoxCeleb2 (xmodal-ft, no FLAG, no Bangla).
# # v7 was: (gender-track experiments, same architecture as v6). v6 was: v5 + paired 7-fold VoxCeleb2/FLAG training, 80% same-gender batches, same-gender hard negatives,
# # longer VoxCeleb2 pre-training, language-adversarial voice tower (no Bangla used), gender-matched / centred score normalization
#
# * Face: VGGFace2 (InceptionResnetV1) features of RetinaFace crops; ViT age band + ViT gender as attributes.
# * Voice: ECAPA-TDNN + WavLM-SV, and the WavLM encoders fine-tuned for face-voice matching (if attached).
# * Model: the same two-tower MLP + cosine recipe (VoxCeleb2 pre-training, FLAG fine-tuning, gender/age heads).
# * Training: 7-fold split over the 70 FLAG identities. Every member x seed is fine-tuned 7 times (each fold held out once);
#   the 7 fold models are averaged at test time, and their out-of-fold scores choose the Stage-A length, the
#   ensemble weights and the System-A settings (all 70 ids are used for validation, 10 at a time).
# * No Bangla data in training (FLAG English training set only; Bengali-speaking VoxCeleb2 speakers removed by language ID).
# * Outputs: System A (test-list clustering, no trial structure) and P (each pair scored on its own).
# * Score sign: the official scorer uses roc_curve(gt, -score) -> files are written negated (LOWER = same person).

# %%
import os, re, glob, json, math, copy, random, shutil, zipfile, time
from collections import defaultdict, Counter
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_curve, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from scipy.special import logsumexp

INPUT_ROOT = os.environ.get("FLAG_INPUT_ROOT", "/kaggle/input")
WORK = os.environ.get("FLAG_WORK", "/kaggle/working")
PHASE = os.environ.get("PHASE", "dev")               # v10: "eval" in the evaluation phase
FACE_SRC = os.environ.get("FACE_SRC", "vgg")         # vgg | arc | arcvgg   (features fed to the face tower)
ATTR_SRC = os.environ.get("ATTR_SRC", "vit")         # vit | insight        (age / gender attributes)
K_FOLDS = 7
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
MEMBERS = [
    dict(name="ew_aux", face=("face",), voice=("ecapa", "wavlm"), face_attr=True, hidden=256, d=128, p_in=0.3, drop=0.5, loss="aux"),
    dict(name="wavlm_ce", face=("face",), voice=("wavlm",), face_attr=True, hidden=512, loss="ce_mse"),
    dict(name="ftL4_aux", face=("face",), voice=("ecapa", "wavlm_ft_L4"), face_attr=True, hidden=256, d=128, p_in=0.3, drop=0.5, loss="aux"),
    # encoder-zoo members (kept only if the features are attached)
    dict(name="vb2_aux", face=("face",), voice=("vb2_ms", "wavlm_ms"), face_attr=True, hidden=256, d=128, p_in=0.3, drop=0.5, loss="aux"),
    dict(name="clip_aux", face=("face", "clipface"), voice=("ecapa_ms", "wavlm_ms"), face_attr=True, hidden=256, d=128, p_in=0.3, drop=0.5, loss="aux"),
    dict(name="rdn_aux", face=("face",), voice=("rdn2",), face_attr=True, hidden=256, d=128, p_in=0.3, drop=0.5, loss="aux"),   # v11: ReDimNet2-B3 voice member (else = ew_aux)
    dict(name="arc_aux", face=("arc",), voice=("ecapa", "wavlm"), face_attr=True, hidden=256, d=128, p_in=0.3, drop=0.5, loss="aux"),   # v11: ArcFace face member
]
def _xft_dirs():
    out = {}
    for p in sorted(glob.glob(os.path.join(os.environ.get("FLAG_INPUT_ROOT", "/kaggle/input"), "**", "flag", "train.npz"), recursive=True)):
        if "face_joint" in np.load(p, allow_pickle=True).files: out.setdefault(os.path.basename(os.path.dirname(os.path.dirname(p))), os.path.dirname(os.path.dirname(p)))
    return out
XFT_DIRS = _xft_dirs()
# v10: runs from the PC trainer (xft_full.py) store VoxCeleb2 features per crop index ("crop_clip_idx"), which cannot be joined to the
# crop-cache rows -> they are used on FLAG only (members without VoxCeleb2 Stage A, i.e. xj0_*); the frozen members keep Stage A.
def _vox_joinable(dd):
    ps = sorted(glob.glob(f"{dd}/vox/*.npz"))
    return bool(ps) and "crop_clip" in np.load(ps[0], allow_pickle=True).files
XJ_DIRS = {V: dd for V, dd in XFT_DIRS.items() if _vox_joinable(dd)}
for _V in XFT_DIRS:                 # fine-tuned encoders (joint embeddings) as tower inputs
    if _V in XJ_DIRS: MEMBERS.append(dict(name=f"xj_{_V}", face=(f"xf_{_V}",), voice=(f"xv_{_V}",), face_attr=True, hidden=256, d=128, p_in=0.3, drop=0.5, loss="aux"))
    # same, but WITHOUT the VoxCeleb2 tower pre-training: the fine-tuned encoders were trained on those very VoxCeleb2 speakers,
    # so their joint embeddings are already aligned there and Stage A has little left to teach (v8 run 1: xj member worse than ew_aux)
    MEMBERS.append(dict(name=f"xj0_{_V}", face=(f"xf_{_V}",), voice=(f"xv_{_V}",), face_attr=True, hidden=256, d=128, p_in=0.3, drop=0.5, loss="aux", no_stage_a=True))
MEMBERS = [m for m in MEMBERS if m["name"] in os.environ.get("MEMBERS", "ew_aux," + ",".join(f"xj_{v},xj0_{v}" for v in XFT_DIRS)).split(",")]
HN_W, HN_MARGIN = float(os.environ.get("HN_W", 0.5)), 0.2      # same-gender hard-negative margin loss
NWAY_W = float(os.environ.get("NWAY_W", 0.0))                     # v9: same-gender N-way matching loss weight (0 = off)
NWAY_N = int(os.environ.get("NWAY_N", 10))                        # N candidates per anchor (1 own identity + N-1 same-gender others); <=0: all
LANG_ADV = float(os.environ.get("LANG_ADV", 0.2))                 # language-adversarial weight on the voice tower (VoxCeleb2 only)
LANG_BAL = os.environ.get("LANG_BAL", "1") == "1"
B_STEP_SCALE = float(os.environ.get("B_STEP_SCALE", "1.0"))      # paper exp 1: scales FLAG-stage steps (item-visit matching); 1.0 = unchanged
DEMO_A_BANDS = int(os.environ.get("DEMO_A_BANDS", "0"))          # paper exp 2: Stage-A single-gender batches also share an age band (0 = off)
QUEUE_A = int(os.environ.get("QUEUE_A", "0"))                     # queue exp 2: Stage-A memory queue size (0 = off); training only
QUEUE_K, QUEUE_W, QUEUE_WARM = int(os.environ.get("QUEUE_K", "16")), float(os.environ.get("QUEUE_W", "0.5")), int(os.environ.get("QUEUE_WARM", "3"))
PROTO_A_W = float(os.environ.get("PROTO_A_W", "0"))               # queue exp 3: Stage-A AAM with SEPARATE face / voice identity prototypes (0 = off)
CLIP_DIVERSE = os.environ.get("CLIP_DIVERSE", "0") == "1"       # queue exp 1: the K items of an identity come from K different clips/audio files
VBANK_P = float(os.environ.get("VBANK_P", "0"))                   # paper exp 3: FLAG-stage prob. of a random 3 s window voice embedding (0 = off)                 # language-balanced speaker sampling in VoxCeleb2
# v11 COMPLIANT (FLAG 2027 FAQ #1-#4): every trial scored on its own -> no per-list centring, no AS-/Z-/T-norm, no per-list z-scoring,
# no clustering of test inputs (System A). Score = fixed-weight sum of plain cosine similarities of the trained towers.
COMPLIANT = os.environ.get("COMPLIANT", "1") == "1"
NORM = "base" if COMPLIANT else os.environ.get("NORM", "center")                           # base | center  (per-list centring of tower outputs before scoring)
AUX_W = 1.0
AUX_GW = 1.0                      # weight of the shared gender head (E5); age head keeps AUX_W
HN_MODE = "any"                   # hard negatives: "off" | "any" (v6: hardest wrong identity) | "samegender" (hardest same-gender wrong identities)
HN_TOPK = 3                       # same-gender mode: average of the k hardest same-gender negatives
SAMPLING = "samegender"           # "samegender" (SAME_GENDER_FRAC of batches single-gender, v6) | "balanced" (half male / half female ids per batch)
TEMP = None                       # None = learnable temperature (v6, starts at 0.07); else a fixed temperature (E4)
USE_AS_NORM, AS_NORM_TOPK = (not COMPLIANT), 300
EMB_DIM, P_IDS, K_PER_ID, SAME_GENDER_FRAC, NOISE_STD, WEIGHT_DECAY = 256, 64, 4, float(os.environ.get("SAME_G", 0.8)), 0.02, 1e-4
A_EPOCHS, A_LR, A_DROPOUT, A_VAL_FRAC = int(os.environ.get("A_EPOCHS", 16)), 1e-3, 0.2, 0.05
A_CKPTS = tuple(int(x) for x in os.environ.get("A_CKPTS", "6,8,12").split(","))
B_EPOCHS, B_LR, B_DROPOUT, B_WARM_EPOCHS = 15, 1e-4, 0.4, 3
MSE_W = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(f"{WORK}/models", exist_ok=True); T0 = time.time()
print("device:", DEVICE, "| face:", FACE_SRC, "| attributes:", ATTR_SRC, "| folds:", K_FOLDS, "| seeds:", SEEDS)

def find(pattern):
    return sorted(glob.glob(os.path.join(INPUT_ROOT, "**", pattern), recursive=True))
def l2n(x):
    x = np.asarray(x, np.float32); return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)
def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr)); return 100 * (fpr[i] + fnr[i]) / 2
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def to_t(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)

# %%
# ================= DATA =================
def subset(D, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(D["face"]) else v) for k, v in D.items()}

FT_DIRS = {os.path.basename(os.path.dirname(p))[3:]: os.path.dirname(p) for p in find("ft_*/flag_train.npz")}
print("fine-tuned encoders attached:", sorted(FT_DIRS))
FLAG_DIR = os.path.dirname([p for p in find("flag/train.npz") if "face_arc" in np.load(p, allow_pickle=True).files][0]); VGG_DIR = os.path.dirname(find("flag_vgg/train.npz")[0])
ZOO_DIR = os.path.dirname(find("zoo/flag/train.npz")[0]) if find("zoo/flag/train.npz") else None
RDN_DIR = os.path.dirname(find("rdn/flag/train.npz")[0]) if find("rdn/flag/train.npz") else None   # v11: ReDimNet2-B3 (frozen) voice embeddings
print("ReDimNet2-B3 features:", RDN_DIR or "not attached")
print("encoder zoo (CLIP / VoxBlink2 / multi-segment):", ZOO_DIR or "not attached")
print("FLAG features:", FLAG_DIR, "| VGGFace2 features:", VGG_DIR)
print("fine-tuned encoder runs (xmodal-ft):", {k: v for k, v in XFT_DIRS.items()})

def pick_face(arc, vgg):
    return {"arc": l2n(arc), "vgg": l2n(vgg), "arcvgg": l2n(np.concatenate([l2n(arc), l2n(vgg)], 1))}[FACE_SRC]

def flag_set(name, has_spk):
    d = np.load(f"{FLAG_DIR}/{name}", allow_pickle=True); v = np.load(f"{VGG_DIR}/{name}", allow_pickle=True)
    n = len(d["face_arc"]); assert len(v["face_vgg"]) == n, name
    if has_spk:
        assert (np.array([s.lower() for s in v["spk"]]) == np.array([s.lower() for s in d["spk"]])).all()
    S = {"row": np.arange(n), "face": pick_face(d["face_arc"], v["face_vgg"]), "arc": l2n(d["face_arc"]), "vgg": l2n(v["face_vgg"]),
         "ecapa": l2n(d["ecapa"]), "wavlm": l2n(d["wavlm"]),
         "age": (v["vit_age"] if ATTR_SRC == "vit" else d["age"]).astype(np.float32),
         "male": (v["vit_male"] if ATTR_SRC == "vit" else d["male"]).astype(np.float32)}
    if RDN_DIR:
        r_ = np.load(f"{RDN_DIR}/{name}", allow_pickle=True); assert len(r_["rdn2"]) == n, name
        if "pid" in r_.files and "pid" in d.files: assert (r_["pid"] == d["pid"]).all(), name
        S["rdn2"] = l2n(r_["rdn2"])
    if ZOO_DIR:
        z = np.load(f"{ZOO_DIR}/{name}", allow_pickle=True); assert len(z["clip_face"]) == n
        S["clipface"] = l2n(z["clip_face"])
        for k in ("vb2", "vb2_ms", "ecapa_ms", "wavlm_ms"):
            S[k] = l2n(z[k])
    for V, dd in XFT_DIRS.items():
        e = np.load(f"{dd}/flag/{name}", allow_pickle=True); assert len(e["face_joint"]) == n, (V, name)
        if has_spk: assert (np.array([str(s).lower() for s in e["spk"]]) == np.array([s.lower() for s in d["spk"]])).all()
        S[f"xf_{V}"], S[f"xv_{V}"] = l2n(e["face_joint"]), l2n(e["voice_joint"])
    ftn = "flag_train.npz" if name == "train.npz" else name
    for L, dd in FT_DIRS.items():
        e = np.load(f"{dd}/{ftn}", allow_pickle=True); assert len(e["wavlm_ft"]) == n
        S[f"wavlm_ft_{L}"] = l2n(e["wavlm_ft"])
    if has_spk:
        S["spk"] = np.array([s.lower() for s in d["spk"]])
    return S

FL = flag_set("train.npz", True)
print("FLAG train:", FL["face"].shape, "| ids:", len(set(FL["spk"])))
test_paths = sorted(glob.glob(f"{FLAG_DIR}/{PHASE}__*.npz"))
print("test sets:", [os.path.basename(p) for p in test_paths])

EX = find("langid/bengali_speakers.json")
EXCL = set(s.lower() for s in json.load(open(EX[0]))["bengali_speakers"]) if EX else set()
print("language ID:", "attached" if EX else "NOT attached", "| Bengali speakers excluded:", len(EXCL))
VOX = None; vox_files = sorted(find("vox/vox2_*/meta.npz"))
LANGS = ["en", "de", "fr", "it", "hi", "es", "nl", "other"]
LG = find("langid/clip_lang.npz")
LANG_OF = {}
if LG:
    lg = np.load(LG[0], allow_pickle=True)
    for t, c, l in zip(lg["tag"], lg["ci"], lg["lang"]):
        code = str(l).split(":")[0].strip(); LANG_OF[(str(t), int(c))] = LANGS.index(code) if code in LANGS else LANGS.index("other")
print("VoxCeleb2 language labels:", len(LANG_OF), "clips")
if vox_files:
    ftv = {L: np.load(f"{dd}/vox.npz", allow_pickle=True) for L, dd in FT_DIRS.items()}
    ftmap = {L: {(str(t), int(c)): i for i, (t, c) in enumerate(zip(e["part_tag"], e["ci"]))} for L, e in ftv.items()}
    parts = defaultdict(list); n_clips = 0
    for p in vox_files:
        d = np.load(p, allow_pickle=True); tag = os.path.basename(os.path.dirname(p))
        vg = find(f"vox_vgg/{tag}/vgg.npz")
        if len(d["clip"]) == 0 or not vg:
            print(f"  {tag}: skipped ({'no VGGFace2 features' if not vg else 'empty'})"); continue
        v = np.load(vg[0], allow_pickle=True)
        cidx = {str(c): i for i, c in enumerate(d["clip"])}
        spk_l = np.array([s.lower() for s in d["spk"]])
        # clip-level ArcFace mean (frames from the voice extraction) for the arc / arcvgg options
        nC = len(d["clip"]); A = np.zeros((nC, 512), np.float32); np.add.at(A, d["face_clip"].astype(int), d["face"]); A = l2n(A)
        rows = [j for j in range(len(v["clip"])) if str(v["clip"][j]) in cidx and v["status"][j] != "nodet"
                and spk_l[cidx[str(v["clip"][j])]] not in EXCL]
        ci = np.array([cidx[str(v["clip"][j])] for j in rows], int)
        # frame index within its clip -> lets the CLIP crops (same clips / frames) be joined to the VGGFace2 rows
        fj, seen = [], defaultdict(int)
        for j in range(len(v["clip"])):
            fj.append(seen[str(v["clip"][j])]); seen[str(v["clip"][j])] += 1
        fj = np.array(fj)
        if FT_DIRS and any((tag, 0) not in ftmap[L] and len(ci) for L in FT_DIRS):
            pass
        keep = np.ones(len(ci), bool)
        for L in FT_DIRS:
            keep &= np.array([(tag, int(c)) in ftmap[L] for c in ci], bool)
        rows = [r for r, k in zip(rows, keep) if k]; ci = ci[keep]
        if RDN_DIR:
            rf_ = find(f"rdn/vox/{tag}.npz")
            if not rf_:
                print(f"  {tag}: no ReDimNet2 features -> skipped"); continue
            rv_ = np.load(rf_[0], allow_pickle=True); mr_ = {str(c): i for i, c in enumerate(rv_["meta_clip"])}
        if ZOO_DIR:
            zf_ = find(f"zoo/vox/{tag}.npz")
            if not zf_:
                print(f"  {tag}: no zoo features -> skipped"); continue
            z = np.load(zf_[0], allow_pickle=True)
            cj, seen = {}, defaultdict(int)
            for r_, c_ in enumerate(z["crop_clip"]):
                cj[(str(c_), seen[str(c_)])] = r_; seen[str(c_)] += 1
            zrow = np.array([cj.get((str(v["clip"][r]), int(fj[r])), -1) for r in rows])
            ok_ = zrow >= 0; rows = [r for r, k in zip(rows, ok_) if k]; ci = ci[ok_]; zrow = zrow[ok_]
            parts["clipface"].append(l2n(z["clip_face"][zrow]))
            mz = {str(c): i for i, c in enumerate(z["meta_clip"])}; zc = np.array([mz[str(d["clip"][c])] for c in ci])
            for k in ("vb2", "vb2_ms", "ecapa_ms", "wavlm_ms"):
                parts[k].append(l2n(z[k][zc]))
        if XJ_DIRS:                     # join the fine-tuned-encoder features: same crop order as crop-cache (clip, frame #), voices per clip
            if not all(os.path.exists(f"{dd}/vox/{tag}.npz") for dd in XJ_DIRS.values()):
                print(f"  {tag}: no fine-tuned-encoder features -> skipped"); continue
            ZX, keep = {}, np.ones(len(rows), bool)
            for V, dd in XJ_DIRS.items():
                z = np.load(f"{dd}/vox/{tag}.npz", allow_pickle=True); cj, seen = {}, defaultdict(int)
                for r_, c_ in enumerate(z["crop_clip"]):
                    cj[(str(c_), seen[str(c_)])] = r_; seen[str(c_)] += 1
                zrow = np.array([cj.get((str(v["clip"][r]), int(fj[r])), -1) for r in rows], int); keep &= zrow >= 0
                ZX[V] = (z, zrow, {str(c): i for i, c in enumerate(z["meta_clip"])})
            rows = [r for r, k in zip(rows, keep) if k]; ci = ci[keep]
            for V, (z, zrow, mz) in ZX.items():
                parts[f"xf_{V}"].append(l2n(z["face_joint"][zrow[keep]].astype(np.float32)))
                parts[f"xv_{V}"].append(l2n(z["voice_joint"][np.array([mz[str(d["clip"][c])] for c in ci], int)].astype(np.float32)))
        parts["lang"].append(np.array([LANG_OF.get((tag, int(c)), -1) for c in ci], np.int64))
        parts["face"].append(pick_face(A[ci], v["face_vgg"][rows])); parts["arc"].append(A[ci]); parts["vgg"].append(l2n(v["face_vgg"][rows]))
        parts["age"].append((v["vit_age"][rows] if ATTR_SRC == "vit" else np.array([d["age"][d["face_clip"] == c].mean() for c in ci])).astype(np.float32))
        parts["male"].append((v["vit_male"][rows] if ATTR_SRC == "vit" else np.array([d["male"][d["face_clip"] == c].mean() for c in ci])).astype(np.float32))
        parts["ecapa"].append(d["ecapa"][ci]); parts["wavlm"].append(d["wavlm"][ci])
        if RDN_DIR: parts["rdn2"].append(rv_["rdn2"][np.array([mr_[str(d["clip"][c])] for c in ci], int)])
        for L, e in ftv.items():
            parts[f"wavlm_ft_{L}"].append(e["wavlm_ft"][[ftmap[L][(tag, int(c))] for c in ci]])
        parts["spk"].append(spk_l[ci]); parts["clip"].append(ci + n_clips); n_clips += nC
        print(f"  {tag}: {len(set(spk_l[ci]))} speakers, {len(set(ci))} clips, {len(ci)} face frames")
    VOX = {k: np.concatenate(v) for k, v in parts.items()}
    for k in ["ecapa", "wavlm"] + [f"wavlm_ft_{L}" for L in FT_DIRS] + (["rdn2"] if RDN_DIR else []):
        VOX[k] = l2n(VOX[k])
    print(f"VoxCeleb2: {len(set(VOX['spk']))} speakers, {len(np.unique(VOX['clip']))} clips, {len(VOX['face'])} frames")

# %%
# ================= GENDER =================
def parse_gender_table(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None, dtype=str)
    except Exception:
        return {}
    df = df.apply(lambda c: c.astype(str).str.strip().str.lower())
    idc = next((c for c in df.columns if df[c].str.match(r"^id\d+$").mean() > 0.6), None)
    gc = next((c for c in df.columns if df[c].isin(["m", "f", "male", "female"]).mean() > 0.6), None)
    if idc is None or gc is None:
        return {}
    return {i: g[0] for i, g in zip(df[idc], df[gc]) if re.match(r"^id\d+$", i) and g in ("m", "f", "male", "female")}

flag_gender = {}
for p in find("*meta*.csv"):
    t = parse_gender_table(p)
    if len(set(FL["spk"]) & set(t)) > len(set(FL["spk"]) & set(flag_gender)):
        flag_gender = t
print("FLAG gender coverage:", len(set(FL["spk"]) & set(flag_gender)), "/", len(set(FL["spk"])))
vox_gender = {}
if VOX is not None:   # majority vote of InsightFace per-frame gender
    acc = defaultdict(list)
    for s, m in zip(VOX["spk"], VOX["male"]):
        if m >= 0:
            acc[s].append(m)
    vox_gender = {s: ("m" if np.mean(v) >= 0.5 else "f") for s, v in acc.items() if v}
    print("VoxCeleb2 gender (InsightFace majority):", Counter(vox_gender.values()))

def speaker_age(S):
    acc = defaultdict(list)
    for s, a in zip(S["spk"], S["age"]):
        if np.isfinite(a):
            acc[s].append(a)
    return {s: float(np.mean(v)) for s, v in acc.items()}
flag_age = speaker_age(FL)
vox_age = speaker_age(VOX) if VOX is not None else {}

# %%
# ================= FEATURES PER MEMBER =================
def face_in(mem, S):
    X = np.concatenate([S[k] for k in mem.get("face", ("face",))], 1)
    if mem["face_attr"] == "age":        # Step 0: frozen ViT age only - predicted gender is NOT a model input
        age = np.nan_to_num((S["age"] - 35.0) / 15.0, nan=0.0)
        X = np.concatenate([X, age[:, None]], 1)
    elif mem["face_attr"]:                # previous behaviour: predicted age AND predicted gender appended to the face input
        age = np.nan_to_num((S["age"] - 35.0) / 15.0, nan=0.0)
        male = np.where(S["male"] < 0, 0.0, S["male"] * 2 - 1)
        X = np.concatenate([X, age[:, None], male[:, None]], 1)
    return X.astype(np.float32)

def voice_in(mem, S):
    return np.concatenate([S[k] for k in mem["voice"]], 1).astype(np.float32)

MEMBERS = [m for m in MEMBERS if all(FL.get(k) is not None for k in m["voice"] + m.get("face", ("face",)))
           and (VOX is None or m.get("no_stage_a") or all(k in VOX for k in m["voice"] + m.get("face", ("face",))))]   # v10: FLAG-only members need no VoxCeleb2 features
print("members:", [m["name"] for m in MEMBERS])

# %%
# ================= MODEL + LOSSES =================
class InputDropout(nn.Dropout):   # fixed rate, not changed by set_dropout()
    pass

class Tower(nn.Module):
    def __init__(self, d_in, d, h, p, p_in=0.0):
        super().__init__()
        self.drop_in = InputDropout(p_in)
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, d))
        self.skip = nn.Linear(d_in, d)
    def forward(self, x):
        x = self.drop_in(x)
        return F.normalize(self.net(x) + self.skip(x), dim=-1)

class FaceVoiceNet(nn.Module):
    def __init__(self, d_face, d_voice, n_ids, h, d=EMB_DIM, p_in=0.0):
        super().__init__()
        self.face, self.voice = Tower(d_face, d, h, A_DROPOUT, p_in), Tower(d_voice, d, h, A_DROPOUT, p_in)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.protos = nn.Parameter(torch.randn(n_ids, d) * 0.01)
        self.head_g, self.head_a = nn.Linear(d, 1), nn.Linear(d, 1)   # shared gender / age heads (both towers)
        self.head_lang = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, len(LANGS)))   # language adversary (voice tower)
        if PROTO_A_W > 0:   # queue exp 3: separate voice prototypes (face uses self.protos); own generator -> global RNG untouched; training only
            self.protos_v = nn.Parameter(torch.randn(n_ids, d, generator=torch.Generator().manual_seed(n_ids)) * 0.01)
    def reset_head(self, n_ids):
        self.protos = nn.Parameter(torch.randn(n_ids, self.protos.shape[1], device=self.protos.device) * 0.01)
        if hasattr(self, "protos_v"):
            self.protos_v = nn.Parameter((torch.randn(n_ids, self.protos.shape[1], generator=torch.Generator().manual_seed(n_ids)) * 0.01).to(self.protos.device))
    def set_dropout(self, p):
        for m in self.modules():
            if isinstance(m, nn.Dropout) and not isinstance(m, InputDropout):
                m.p = p
    def set_towers_trainable(self, flag):
        for t in (self.face, self.voice):
            for prm in t.parameters():
                prm.requires_grad = flag

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -g

def hard_negative_samegender(zf, zv, y, g, margin=HN_MARGIN, k=None):
    """Same-gender hard negatives (training-time mask only): for every face, the k most similar voices of OTHER identities
    with the SAME gender must score `margin` below the face's own-identity voices (and vice versa). Gender never enters the
    embeddings or the score."""
    k = k or HN_TOPK
    s = zf @ zv.T; pos = (y[:, None] == y[None, :]); same_g = (g[:, None] == g[None, :]) & (g[:, None] >= 0)
    neg = (~pos) & same_g
    if neg.sum() == 0:
        return s.new_zeros(())
    sp = (s * pos.float()).sum(1) / pos.float().sum(1); sp2 = (s * pos.float()).sum(0) / pos.float().sum(0)
    def topk_mean(mat, mask, dim):
        m = mat.masked_fill(~mask, -4.0); kk = min(k, mat.shape[dim])
        v = m.topk(kk, dim=dim).values; valid = (v > -3.9).float()
        return (v * valid).sum(dim) / valid.sum(dim).clamp(min=1), valid.sum(dim) > 0
    n1, ok1 = topk_mean(s, neg, 1); n2, ok2 = topk_mean(s, neg, 0)
    l1 = F.relu(margin + n1 - sp)[ok1].mean() if ok1.any() else s.new_zeros(())
    l2 = F.relu(margin + n2 - sp2)[ok2].mean() if ok2.any() else s.new_zeros(())
    return 0.5 * (l1 + l2)

HN_STATS = []      # Phase 1 (C): per-batch log of hard-negative selection (count, mean selected-negative cosine, mean positive cosine)
def hard_negative_semihard(zf, zv, y, margin=HN_MARGIN):
    """Phase 1 (C1): semi-hard negatives. For every face (and every voice): the most similar wrong-identity partner whose
    similarity is still BELOW its own-identity average (s_neg < s_pos) but within `margin` of it; the margin loss then pushes it
    further below. Anchors without such a negative contribute nothing. Identity labels only - gender is never used."""
    s = zf @ zv.T; pos = (y[:, None] == y[None, :]).float()
    def side(s, pos):
        sp = (s * pos).sum(1) / pos.sum(1)
        band = (pos == 0) & (s < sp[:, None]) & (s > sp[:, None] - margin)
        sn = s.masked_fill(~band, -4.0).max(1).values; ok = band.any(1)
        with torch.no_grad():
            HN_STATS.append((int(ok.sum()), float(sn[ok].mean()) if ok.any() else float("nan"), float(sp.mean())))
        return F.relu(margin + sn - sp)[ok].mean() if ok.any() else s.new_zeros(())
    return 0.5 * (side(s, pos) + side(s.T, pos.T))

def hard_negative(zf, zv, y, margin=HN_MARGIN):
    """For every face (and every voice): the hardest wrong-identity partner in the batch must score `margin` below
    its own identity's average. Batches are mostly same-gender, so these are same-gender hard negatives."""
    s = zf @ zv.T; pos = (y[:, None] == y[None, :]).float()
    sp = (s * pos).sum(1) / pos.sum(1); sn = (s - 4 * pos).max(1).values
    sp2 = (s * pos).sum(0) / pos.sum(0); sn2 = (s - 4 * pos).max(0).values
    with torch.no_grad():
        HN_STATS.append((int((margin + sn - sp > 0).sum()), float(sn.mean()), float(sp.mean())))
    return 0.5 * (F.relu(margin + sn - sp).mean() + F.relu(margin + sn2 - sp2).mean())

def nway_samegender(zf, zv, y, g, scale, n):
    """v9: N-way cross-modal matching with SAME-GENDER distractors (training only). Every voice must pick a face of its own
    identity out of its own-identity faces + (n-1) randomly drawn faces of OTHER identities with the SAME gender (and every face
    the matching voice, symmetrically). n <= 0: all same-gender faces in the batch. Gender only chooses the distractors;
    it never enters the embeddings or the score."""
    s = scale * zf @ zv.T; pos = y[:, None] == y[None, :]
    neg = (~pos) & (g[:, None] == g[None, :]) & (g[:, None] >= 0)
    def side(l, pos, neg):
        if n > 0:
            r = torch.rand_like(l).masked_fill(~neg, -1.0)
            thr = r.topk(min(n - 1, l.shape[1]), dim=1).values[:, -1:]
            neg = neg & (r >= thr)
        ok = neg.any(1)
        if not ok.any():
            return l.new_zeros(())
        l = l.masked_fill(~(pos | neg), float("-inf")); logp = l - torch.logsumexp(l, 1, keepdim=True)
        p = pos.float()
        return -(logp.masked_fill(~pos, 0.0) * p).sum(1).div(p.sum(1))[ok].mean()
    return 0.5 * (side(s, pos, neg) + side(s.T, pos.T, neg.T))

PROTO_ON = False    # True only inside Stage A when PROTO_A_W > 0
QUEUE = None        # {"f","v": detached past embeddings, "y": labels, "n": filled, "ptr", "active"}; set only inside Stage A when QUEUE_A > 0
QSTATS = []
def queue_term(zf, zv, y, margin=HN_MARGIN):
    """Queue exp 2 (training only): each face's QUEUE_K hardest past VOICES of other identities (and each voice's hardest past faces)
    must score `margin` below the anchor's own-identity batch average - the batch-hard margin objective extended to a memory queue.
    SupCon is untouched; the queue is never used at validation or inference."""
    n = QUEUE["n"]; qf, qv, qy = QUEUE["f"][:n].clone(), QUEUE["v"][:n].clone(), QUEUE["y"][:n].clone()   # clones: the queue is overwritten before backward
    s = zf @ zv.T; pos = (y[:, None] == y[None, :]).float(); sp = (s * pos).sum(1) / pos.sum(1); sp2 = (s * pos).sum(0) / pos.sum(0)
    k = min(QUEUE_K, n)
    sf = (zf @ qv.T).masked_fill(y[:, None] == qy[None, :], -4.0).topk(k, dim=1).values      # faces vs queued voices
    sv = (zv @ qf.T).masked_fill(y[:, None] == qy[None, :], -4.0).topk(k, dim=1).values      # voices vs queued faces
    with torch.no_grad():
        QSTATS.append((float(sf.mean()), float((margin + sf - sp[:, None] > 0).float().mean())))
    return 0.5 * (F.relu(margin + sf - sp[:, None]).mean() + F.relu(margin + sv - sp2[:, None]).mean())
def queue_push(zf, zv, y):
    b = len(y); i = (QUEUE["ptr"] + torch.arange(b, device=y.device)) % QUEUE["f"].shape[0]
    QUEUE["f"][i], QUEUE["v"][i], QUEUE["y"][i] = zf.detach(), zv.detach(), y
    QUEUE["ptr"] = int((QUEUE["ptr"] + b) % QUEUE["f"].shape[0]); QUEUE["n"] = min(QUEUE["n"] + b, QUEUE["f"].shape[0])

def supcon_xmodal(zf, zv, y, scale):
    logits = scale * zf @ zv.T; pos = (y[:, None] == y[None, :]).float()
    def side(l):
        logp = l - torch.logsumexp(l, 1, keepdim=True); return -(logp * pos).sum(1).div(pos.sum(1)).mean()
    return 0.5 * (side(logits) + side(logits.T))

def margin_softmax(z, protos, y, m, s=30.0):
    cos = z @ F.normalize(protos, dim=-1).T
    if m:
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        cos = torch.where(F.one_hot(y, protos.shape[0]).bool(), torch.cos(theta + m), cos)
    return F.cross_entropy(s * cos, y)

def total_loss(mem, model, f, v, y, g=None, a=None, lng=None):
    zf, zv = model.face(f), model.voice(v)
    scale = model.logit_scale.exp().clamp(max=100) if TEMP is None else torch.tensor(1.0 / TEMP, device=zf.device)
    L = supcon_xmodal(zf, zv, y, scale)
    if NWAY_W and g is not None:
        L = L + NWAY_W * nway_samegender(zf, zv, y, g, scale, NWAY_N)
    if HN_W and HN_MODE == "any":
        L = L + HN_W * hard_negative(zf, zv, y)
    elif HN_W and HN_MODE == "semihard":
        L = L + HN_W * hard_negative_semihard(zf, zv, y)
    elif HN_W and HN_MODE == "samegender" and g is not None:
        L = L + HN_W * hard_negative_samegender(zf, zv, y, g)
    if PROTO_ON:            # queue exp 3 (Stage A only): AAM (margin 0.2, scale 30) of faces vs face prototypes and voices vs voice prototypes
        L = L + PROTO_A_W * 0.5 * (margin_softmax(zf, model.protos, y, 0.2) + margin_softmax(zv, model.protos_v, y, 0.2))
    if QUEUE is not None:
        if QUEUE["active"] and QUEUE["n"] > 0:
            L = L + QUEUE_W * queue_term(zf, zv, y)
        queue_push(zf, zv, y)
    if LANG_ADV and lng is not None:
        ok = lng >= 0
        if ok.sum() > 1:
            L = L + LANG_ADV * F.cross_entropy(model.head_lang(_GRL.apply(zv[ok])), lng[ok])
    if mem["loss"] in ("aam", "ce_mse"):
        m = 0.2 if mem["loss"] == "aam" else 0.0
        L = L + 0.5 * (margin_softmax(zf, model.protos, y, m) + margin_softmax(zv, model.protos, y, m))
    if mem["loss"] == "ce_mse":
        L = L + MSE_W * ((zf - zv) ** 2).sum(1).mean()
    if mem["loss"] == "aux" and g is not None:
        ok = g >= 0
        for z in (zf, zv):
            if ok.any():
                L = L + AUX_GW * F.binary_cross_entropy_with_logits(model.head_g(z[ok]).squeeze(1), g[ok])
            L = L + AUX_W * F.smooth_l1_loss(model.head_a(z).squeeze(1), a)
    return L

# %%
# ================= SAMPLER / TRIALS / EVAL =================
class PKSampler:
    def __init__(self, y, gender_of, seed, lang=None, age_of=None, bands=0, clip=None):
        self.rng = np.random.default_rng(seed); self.by_id = defaultdict(list)
        for i, c in enumerate(y):
            self.by_id[c].append(i)
        self.by_clip = None
        if CLIP_DIVERSE and clip is not None:   # identity -> list of item groups, one group per clip / audio file
            self.by_clip = {}
            for c, items in self.by_id.items():
                g_ = defaultdict(list)
                for i in items: g_[clip[i]].append(i)
                self.by_clip[c] = list(g_.values())
            nc = np.array([len(v) for v in self.by_clip.values()])
            print(f"  clip-diverse sampling: clips per identity median {np.median(nc):.0f}, identities with < {K_PER_ID} clips: {(nc < K_PER_ID).sum()} / {len(nc)}", flush=True)
        self.ids = np.array(sorted(self.by_id))
        self.w = None
        if lang is not None and LANG_BAL:        # speakers weighted by 1/sqrt(#speakers of their language)
            lid = {c: Counter(lang[self.by_id[c]]).most_common(1)[0][0] for c in self.ids}
            cnt = Counter(lid.values()); self.w = {c: 1.0 / np.sqrt(cnt[lid[c]]) for c in self.ids}
        self.by_g = {g: np.array([c for c in self.ids if gender_of.get(c) == g]) for g in ("m", "f")}
        self.by_g = {g: v for g, v in self.by_g.items() if len(v) >= 4}
        self.by_gb = {}
        if bands and age_of:                     # within-gender age bands (quantiles of speaker mean ViT age); unknown age -> mixed batches only
            for g, ids_g in self.by_g.items():
                known = np.array([c for c in ids_g if c in age_of]); ages = np.array([age_of[c] for c in known])
                edges = np.quantile(ages, [i / bands for i in range(1, bands)]); b_of = np.searchsorted(edges, ages, side="right")
                for b in range(bands):
                    if (b_of == b).sum() >= 4: self.by_gb[(g, b)] = known[b_of == b]
            print("  age-band pools:", {f"{g}{b}": (len(v), round(float(np.min([age_of[c] for c in v])), 1), round(float(np.max([age_of[c] for c in v])), 1))
                                        for (g, b), v in sorted(self.by_gb.items())}, flush=True)
    def take(self, c):
        """K items of identity c; clip-diverse: from K different clips (then any remaining items if it has fewer clips)."""
        s = self.by_id[c]
        if self.by_clip is None: return list(self.rng.choice(s, size=min(K_PER_ID, len(s)), replace=False))
        groups = self.by_clip[c]; gi = self.rng.permutation(len(groups))[:K_PER_ID]
        out = [int(self.rng.choice(groups[g])) for g in gi]
        if len(out) < min(K_PER_ID, len(s)):
            rest = [i for i in s if i not in out]; out += list(self.rng.choice(rest, size=min(K_PER_ID, len(s)) - len(out), replace=False))
        return out
    def batch(self):
        pool = self.ids
        if SAMPLING == "balanced" and len(self.by_g) == 2:       # half male, half female identities in every batch
            idx = []
            for gg in ("m", "f"):
                pool_g = self.by_g[gg]; p_g = None
                if self.w is not None:
                    p_g = np.array([self.w[c] for c in pool_g]); p_g = p_g / p_g.sum()
                for c in self.rng.choice(pool_g, size=min(P_IDS // 2, len(pool_g)), replace=False, p=p_g):
                    s_ = self.by_id[c]; idx += list(self.rng.choice(s_, size=min(K_PER_ID, len(s_)), replace=False))
            return np.array(idx)
        if self.by_g and self.rng.random() < SAME_GENDER_FRAC:
            g_ = self.rng.choice(sorted(self.by_g)); pool = self.by_g[g_]
            if self.by_gb:                       # paper exp 2: one age band of that gender, chosen in proportion to its size
                ks = [k for k in sorted(self.by_gb) if k[0] == g_]; sz = np.array([len(self.by_gb[k]) for k in ks], float)
                pool = self.by_gb[ks[self.rng.choice(len(ks), p=sz / sz.sum())]]
        idx = []
        p = None
        if self.w is not None:
            p = np.array([self.w[c] for c in pool]); p = p / p.sum()
        for c in self.rng.choice(pool, size=min(P_IDS, len(pool)), replace=False, p=p):
            idx += self.take(c)
        return np.array(idx)

def make_trials(y, gender_of, seed):
    rng = np.random.default_rng(seed); by_id = defaultdict(list)
    for i, c in enumerate(y):
        by_id[c].append(i)
    ids = sorted(by_id); std, gen = [], []
    for i, c in enumerate(y):
        j = rng.choice([k for k in by_id[c] if k != i] or by_id[c])
        std.append((i, j, 1)); gen.append((i, j, 1))
        std.append((i, rng.choice(by_id[rng.choice([d for d in ids if d != c])]), 0))
        sg = [d for d in ids if d != c and gender_of.get(c) and gender_of.get(d) == gender_of.get(c)]
        if sg:
            gen.append((i, rng.choice(by_id[rng.choice(sg)]), 0))
    return np.array(std), (np.array(gen) if len({t[2] for t in gen}) == 2 else None)

@torch.no_grad()
def project(model, X, which, bs=8192):
    model.eval(); tower = model.face if which == "face" else model.voice
    return torch.cat([tower(X[i:i + bs]) for i in range(0, len(X), bs)])

@torch.no_grad()
def pair_scores(model, Fp, Vp, cohort=None):
    zf, zv = project(model, Fp, "face"), project(model, Vp, "voice"); s = (zf * zv).sum(1)
    if cohort is None:
        return s.cpu().numpy()
    cf, cv = project(model, cohort[0], "face"), project(model, cohort[1], "voice"); k = min(AS_NORM_TOPK, len(cf))
    tf, tv = (zf @ cv.T).topk(k, dim=1).values, (zv @ cf.T).topk(k, dim=1).values
    return (0.5 * ((s - tf.mean(1)) / (tf.std(1) + 1e-6) + (s - tv.mean(1)) / (tv.std(1) + 1e-6))).cpu().numpy()

def trial_eer(model, Fx, Vx, trials, cohort=None):
    if trials is None:
        return float("nan")
    return eer(trials[:, 2], pair_scores(model, Fx[trials[:, 0]], Vx[trials[:, 1]], cohort))

def run_stage(mem, model, Ftr, Vtr, ytr, gender_of, epochs, lr, seed, n_units, val=None, warm=0, tag="", age_of=None, ckpts=(), lang=None, bands=0, vbank=None, clip=None, queue=0, proto=False):
    global QUEUE, PROTO_ON
    PROTO_ON = bool(proto)
    sampler = PKSampler(ytr, gender_of, seed, lang, age_of=age_of, bands=bands, clip=clip)
    QUEUE = None
    if queue:
        d_ = mem.get("d", EMB_DIM); QSTATS.clear()
        QUEUE = {"f": torch.zeros(queue, d_, device=DEVICE), "v": torch.zeros(queue, d_, device=DEVICE),
                 "y": torch.full((queue,), -1, dtype=torch.long, device=DEVICE), "n": 0, "ptr": 0, "active": False}
    l_t = torch.tensor(lang, device=DEVICE) if lang is not None else None
    lab = {c: i for i, c in enumerate(sorted(set(ytr)))}
    y_t = torch.tensor([lab[c] for c in ytr], device=DEVICE)
    g_t = torch.tensor([{"m": 1.0, "f": 0.0}.get(gender_of.get(c), -1.0) for c in ytr], device=DEVICE)
    a_t = torch.tensor([((age_of or {}).get(c, 35.0) - 35.0) / 15.0 for c in ytr], device=DEVICE, dtype=torch.float32)
    steps = max(1, math.ceil(n_units / (P_IDS * K_PER_ID)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total, wsteps = epochs * steps, 2 * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / wsteps) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    best = (float("inf"), None, epochs)
    for ep in range(epochs):
        if QUEUE is not None: QUEUE["active"] = ep >= QUEUE_WARM
        model.set_towers_trainable(ep >= warm); model.train(); losses = []
        for _ in range(steps):
            idx = torch.tensor(sampler.batch(), device=DEVICE); f, v = Ftr[idx], Vtr[idx]
            if vbank is not None:                # paper exp 3: swap some whole-clip voice vectors for one random 3 s window of the same clip
                sw = torch.rand(len(idx), device=DEVICE) < VBANK_P
                j = vbank["lo"][idx] + (torch.rand(len(idx), device=DEVICE) * vbank["n"][idx]).long().clamp(max=vbank["n"][idx] - 1)
                v = torch.where(sw[:, None], vbank["W"][j], v)
            if NOISE_STD:
                f = f + NOISE_STD * torch.randn_like(f); v = v + NOISE_STD * torch.randn_like(v)
            loss = total_loss(mem, model, f, v, y_t[idx], g_t[idx], a_t[idx], l_t[idx] if l_t is not None else None)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); losses.append(loss.item())
        if ep + 1 in ckpts:
            model._ckpts = getattr(model, "_ckpts", {}); model._ckpts[ep + 1] = copy.deepcopy(model.state_dict())
        if val is not None:
            e1, e2 = trial_eer(model, val[0], val[1], val[2]), trial_eer(model, val[0], val[1], val[3])
            sc = np.nanmean([e1, e2])
            if ep >= warm and sc < best[0]:
                best = (sc, copy.deepcopy(model.state_dict()), ep + 1)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  [{tag}] ep {ep+1:3d} loss {np.mean(losses):.3f} | val EER std {e1:5.2f} gender {e2:5.2f}", flush=True)
    model.set_towers_trainable(True)
    PROTO_ON = False
    if QUEUE is not None:
        if QSTATS: print(f"  [{tag}] queue {QUEUE['f'].shape[0]}, top-{QUEUE_K} from epoch {QUEUE_WARM + 1}: {len(QSTATS)} steps, mean top-k cos "
                         f"{np.mean([q[0] for q in QSTATS]):.3f}, margin violations {100 * np.mean([q[1] for q in QSTATS]):.0f}%", flush=True)
        QUEUE = None
    if val is not None and best[1] is not None:
        model.load_state_dict(best[1]); print(f"  [{tag}] best epoch {best[2]} | val EER {best[0]:.2f}")
    return best[2], best[0]

# %%
# ================= CLUSTERING HELPERS =================
def cluster(X, thr):
    if len(X) < 3:
        return np.arange(len(X))
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=thr).fit_predict(X)

def refine(X, lab, min_size):
    """Attach members of small clusters (< min_size) to the nearest large cluster centroid."""
    if min_size <= 1:
        return lab
    sizes = np.bincount(lab); big = np.where(sizes >= min_size)[0]
    if len(big) == 0:
        return lab
    C = l2n(np.stack([X[lab == k].mean(0) for k in big])); lab = lab.copy(); small = ~np.isin(lab, big)
    lab[small] = big[(X[small] @ C.T).argmax(1)]
    return np.unique(lab, return_inverse=True)[1]

def tune_threshold(X, y, grid=np.arange(0.30, 0.96, 0.05)):
    rng = np.random.default_rng(0); ids = np.array(sorted(set(y))); res = defaultdict(list)
    for _ in range(4):
        idx = np.where(np.isin(y, rng.choice(ids, size=min(15, len(ids)), replace=False)))[0]
        idx = rng.choice(idx, size=min(len(idx), 1200), replace=False)
        for t in grid:
            res[t].append(adjusted_rand_score(y[idx], cluster(X[idx], t)))
    best = max(res, key=lambda t: np.mean(res[t])); print(f"  threshold {best:.2f} ARI {np.mean(res[best]):.3f}"); return float(best)

def sinkhorn_log(S, tau, iters=100):
    L = S / tau
    for _ in range(iters):
        L = L - logsumexp(L, axis=1, keepdims=True); L = L - logsumexp(L, axis=0, keepdims=True)
    return L

def cooc_lift(a, b, alpha=1.0):
    N = len(a); C = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(C, (a, b), 1)
    n_a, m_b = C.sum(1), C.sum(0)
    return np.log((C[a, b] - 1 + alpha) / ((n_a[a] - 1) * (m_b[b] - 1) / max(N - 1, 1) + alpha))

def combine(comp, w_clus, w_asg, tau, w_co=0.0):
    pair, S, a, b = comp; s = zs(pair)
    if w_clus:
        s = s + w_clus * zs(S[a, b])
    if w_asg:
        s = s + w_asg * zs(sinkhorn_log(S, tau)[a, b])
    if w_co:
        s = s + w_co * zs(cooc_lift(a, b))
    return s

def sim_pairs(y, gender_of, same_gender, seed):
    rng = np.random.default_rng(seed); by_id = defaultdict(list)
    for i, c in enumerate(y):
        by_id[c].append(i)
    fi, vi, lab = [], [], []
    for i, c in enumerate(y):
        if rng.random() < 0.5:
            fi.append(i); vi.append(i); lab.append(1); continue
        others = [d for d in by_id if d != c and (not same_gender or gender_of.get(d) == gender_of.get(c))]
        if others:
            fi.append(i); vi.append(rng.choice(by_id[rng.choice(others)])); lab.append(0)
    return np.array(fi), np.array(vi), np.array(lab)


# %%
# ================= 7-FOLD SPLITS: FLAG identities and VoxCeleb2 speakers (paired: model k never sees either fold k) =================
rng0 = np.random.default_rng(1234)
flag_ids = np.array(sorted(set(FL["spk"])))
FOLD_OF = {}
for g in ("m", "f", None):                       # gender-stratified: each fold gets ~6 male + ~4 female ids
    pool = [c for c in flag_ids if flag_gender.get(c) == g] if g else [c for c in flag_ids if flag_gender.get(c) not in ("m", "f")]
    pool = list(rng0.permutation(pool))
    for i, c in enumerate(pool):
        FOLD_OF[c] = (i + len(FOLD_OF)) % K_FOLDS
FOLD = np.array([FOLD_OF[s] for s in FL["spk"]])
print("FLAG folds (ids per fold):", dict(sorted(Counter(FOLD_OF.values()).items())))
if VOX is not None:
    vids = np.array(sorted(set(VOX["spk"]))); VFOLD_OF = {c: i % K_FOLDS for i, c in enumerate(rng0.permutation(vids))}
    VFOLD = np.array([VFOLD_OF[s] for s in VOX["spk"]])
    print("VoxCeleb2 folds (speakers per fold):", dict(sorted(Counter(VFOLD_OF.values()).items())))

OOF_TRIALS = {}       # fixed out-of-fold FLAG trials, always within one fold (one model scores both sides)
for k in range(K_FOLDS):
    idx = np.where(FOLD == k)[0]; st, gn = make_trials(FL["spk"][idx], flag_gender, 7 + k)
    OOF_TRIALS[k] = (idx, st, gn)

# %%
# ================= TRAIN: members x seeds x 7 paired folds =================
def finetune(mem, stateA, S, epochs, seed):
    dF, dV = face_in(mem, S).shape[1], voice_in(mem, S).shape[1]
    m = FaceVoiceNet(dF, dV, stateA["protos"].shape[0], mem["hidden"], mem.get("d", EMB_DIM), mem.get("p_in", 0.0)).to(DEVICE)
    m.load_state_dict(stateA); m.reset_head(len(set(S["spk"]))); m.set_dropout(mem.get("drop", B_DROPOUT))
    run_stage(mem, m, to_t(face_in(mem, S)), to_t(voice_in(mem, S)), S["spk"], flag_gender, epochs, B_LR, seed,
              int(round(len(S["spk"]) * B_STEP_SCALE)), warm=B_WARM_EPOCHS, age_of=flag_age, vbank=voice_bank(mem, S), clip=audio_key(S))
    return m

def audio_key(S):
    """FLAG items sharing one audio file have identical ECAPA vectors -> one key per audio file."""
    return np.unique(np.round(S["ecapa"], 5), axis=0, return_inverse=True)[1].ravel()

SEG_BANK = None
def voice_bank(mem, S):
    """Paper exp 3: per-row window-embedding bank for the FLAG training rows of S, in voice_in() layout (each key L2-normalised)."""
    global SEG_BANK
    if VBANK_P <= 0: return None
    if SEG_BANK is None:
        z = np.load(find("seg/flag/train_seg.npz")[0], allow_pickle=True)
        assert len(z["item_u"]) == len(FL["spk"]) and (np.array([str(x).lower() for x in z["spk"]]) == np.array([str(x).lower() for x in FL["spk"]])).all()
        SEG_BANK = {"item_u": z["item_u"], "off": z["off"], "ecapa": l2n(z["ecapa_w"]), "wavlm": l2n(z["wavlm_w"]), "rdn2": l2n(z["rdn2_w"])}
        print(f"  window bank: {len(z['wav'])} clips, {z['off'][-1]} windows", flush=True)
    if not all(k in SEG_BANK for k in mem["voice"]): return None
    u = SEG_BANK["item_u"][S["row"]]; lo, hi = SEG_BANK["off"][u], SEG_BANK["off"][u + 1]
    W = np.concatenate([SEG_BANK[k] for k in mem["voice"]], 1).astype(np.float32)
    return {"W": to_t(W), "lo": torch.tensor(lo, device=DEVICE), "n": torch.tensor(hi - lo, device=DEVICE)}

SCRATCH_EPOCHS, SCRATCH_LR = int(os.environ.get("SCRATCH_EPOCHS", 25)), float(os.environ.get("SCRATCH_LR", 1e-3))
def finetune_scratch(mem, S, seed):
    """FLAG-only tower training (no VoxCeleb2 Stage A) for members with no_stage_a."""
    dF, dV = face_in(mem, S).shape[1], voice_in(mem, S).shape[1]; set_seed(seed)
    m = FaceVoiceNet(dF, dV, len(set(S["spk"])), mem["hidden"], mem.get("d", EMB_DIM), mem.get("p_in", 0.0)).to(DEVICE); m.set_dropout(mem.get("drop", B_DROPOUT))
    run_stage(mem, m, to_t(face_in(mem, S)), to_t(voice_in(mem, S)), S["spk"], flag_gender, SCRATCH_EPOCHS, SCRATCH_LR, seed, int(round(len(S["spk"]) * B_STEP_SCALE)), age_of=flag_age, clip=audio_key(S))
    return m

def oof_eer(m, mem, k):
    idx, st, gn = OOF_TRIALS[k]; VA = subset(FL, idx); Fx, Vx = to_t(face_in(mem, VA)), to_t(voice_in(mem, VA))
    return np.nanmean([trial_eer(m, Fx, Vx, st), trial_eer(m, Fx, Vx, gn)])

def stage_a(mem, k, seed, ckpts):
    """VoxCeleb2 pre-training without VoxCeleb2 fold k (fold k = validation speakers); returns {epoch: state}."""
    dF, dV = face_in(mem, FL).shape[1], voice_in(mem, FL).shape[1]; set_seed(seed * 100 + k)
    tr = np.where(VFOLD != k)[0]; TR = subset(VOX, tr)
    va = np.where(VFOLD == k)[0]; _, first = np.unique(VOX["clip"][va], return_index=True); VA = subset(VOX, va[first])
    vst, vgn = make_trials(VA["spk"], vox_gender, 7)
    m = FaceVoiceNet(dF, dV, len(set(TR["spk"])), mem["hidden"], mem.get("d", EMB_DIM), mem.get("p_in", 0.0)).to(DEVICE)
    m.set_dropout(mem.get("drop", A_DROPOUT))
    run_stage(mem, m, to_t(face_in(mem, TR)), to_t(voice_in(mem, TR)), TR["spk"], vox_gender, max(ckpts), A_LR, seed,
              len(np.unique(TR["clip"])), val=(to_t(face_in(mem, VA)), to_t(voice_in(mem, VA)), vst, vgn), tag=f"A vox fold{k}",
              age_of=vox_age, ckpts=ckpts, lang=TR["lang"] if "lang" in TR else None, bands=DEMO_A_BANDS, clip=TR["clip"], queue=QUEUE_A, proto=PROTO_A_W > 0)
    return m._ckpts

@torch.no_grad()
def member_scores(mem, models, S, fi, vi, cohort_S):
    """Mean over models of z-normed scores: cosine of tower outputs (optionally centred on the list), AS-norm vs the cohort."""
    Fx, Vx = to_t(face_in(mem, S)), to_t(voice_in(mem, S)); ctr = NORM in ("center", "both")
    Cf_, Cv_ = (to_t(face_in(mem, cohort_S)), to_t(voice_in(mem, cohort_S))) if (USE_AS_NORM and cohort_S is not None) else (None, None)
    out = []
    for m in models:
        zf, zv = project(m, Fx, "face"), project(m, Vx, "voice")
        if ctr:
            zf, zv = F.normalize(zf - zf.mean(0), dim=-1), F.normalize(zv - zv.mean(0), dim=-1)
        s_ = (zf[fi] * zv[vi]).sum(1)
        if Cf_ is not None:
            cf, cv = project(m, Cf_, "face"), project(m, Cv_, "voice")
            if ctr:
                cf, cv = F.normalize(cf - cf.mean(0), dim=-1), F.normalize(cv - cv.mean(0), dim=-1)
            kk = min(AS_NORM_TOPK, len(cf)); tf = (zf[fi] @ cv.T).topk(kk, dim=1).values; tv = (zv[vi] @ cf.T).topk(kk, dim=1).values
            s_ = 0.5 * ((s_ - tf.mean(1)) / (tf.std(1) + 1e-6) + (s_ - tv.mean(1)) / (tv.std(1) + 1e-6))
        out.append(s_.cpu().numpy() if COMPLIANT else zs(s_.cpu().numpy()))
    return np.mean(out, 0)


# ================= GENDER-TRACK EXPERIMENTS (same architecture; one member, one seed, Stage A fixed at 8 epochs) =================
# Every experiment: identical 7-fold splits (FLAG ids + VoxCeleb2 speakers), identical held-out trials, identical scoring
# (per-list centring + AS-norm, exactly as in the submission). Primary metric = held-out SAME-GENDER EER (like the gender track);
# a change is kept only if it lowers it without raising the random-negative EER (like no-gender) by more than 0.5.
RUN_EXPERIMENTS = os.environ.get("RUN_EXPERIMENTS", "0") == "1"
EXP_A_EP, EXP_SEED = 8, 0
EXP_MEM = dict([m for m in MEMBERS if m["name"] == "ew_aux"][0])

def supcon_unit_test():
    """E1: check the positive / negative masks and that the losses point the right way on a toy batch."""
    torch.manual_seed(0); y = torch.tensor([0, 0, 1, 1, 2, 2]); g = torch.tensor([1., 1., 0., 0., 1., 1.])
    pos = (y[:, None] == y[None, :])
    assert pos.sum(1).tolist() == [2] * 6, "every sample must have exactly its own identity's 2 voices as positives"
    C = F.normalize(torch.randn(3, 16), dim=-1)                       # one centre per identity
    zf = F.normalize(C[y] + 0.1 * torch.randn(6, 16), dim=-1)         # faces near their identity centre
    zv_aligned = F.normalize(C[y] + 0.1 * torch.randn(6, 16), dim=-1) # voices of the same identities near the same centres
    zv_random = F.normalize(torch.randn(6, 16), dim=-1)
    la, lr_ = supcon_xmodal(zf, zv_aligned, y, 1 / 0.07).item(), supcon_xmodal(zf, zv_random, y, 1 / 0.07).item()
    ha, hr = hard_negative_samegender(zf, zv_aligned, y, g).item(), hard_negative_samegender(zf, zv_random, y, g).item()
    neg_sg = ((~pos) & (g[:, None] == g[None, :])).sum(1).tolist()
    print(f"[E1] positives per sample {pos.sum(1).tolist()} | same-gender negatives per sample {neg_sg}")
    print(f"[E1] SupCon (face->voice + voice->face): aligned {la:.3f} < random {lr_:.3f} | same-gender hard-negative: aligned {ha:.3f} <= random {hr:.3f}")
    assert la < lr_ and ha <= hr
supcon_unit_test()

def cfg_now():
    return dict(HN_W=HN_W, HN_MODE=HN_MODE, SAMPLING=SAMPLING, SAME_GENDER_FRAC=SAME_GENDER_FRAC, TEMP=TEMP, AUX_GW=AUX_GW,
                FACE_ATTR=EXP_MEM["face_attr"], B_EPOCHS=B_EPOCHS, B_LR=B_LR, WEIGHT_DECAY=WEIGHT_DECAY, NWAY_W=NWAY_W, NWAY_N=NWAY_N)
def cfg_apply(cfg):
    for k, v in cfg.items():
        if k == "FACE_ATTR":
            EXP_MEM["face_attr"] = v
        else:
            globals()[k] = v

def write_zip(name, mem_models, dirname):
    """P-style submission (each pair scored on its own) from a list of (member, [fold models])."""
    SUB = f"{WORK}/{dirname}/{name}"; shutil.rmtree(SUB, ignore_errors=True)
    for path in test_paths:
        _, track, lang = os.path.basename(path)[:-4].split("__")
        S = flag_set(os.path.basename(path), False); idx = np.arange(len(S["face"]))
        sc = zs(sum(member_scores(mem, ms, S, idx, idx, FL) for mem, ms in mem_models))
        out = f"{SUB}/{OUT_NAMES_[(track, lang)]}"; os.makedirs(os.path.dirname(out), exist_ok=True)
        pid = np.load(path, allow_pickle=True)["pid"]
        with open(out, "w") as fh:
            for p_, v in zip(pid, sc):
                fh.write(f"{p_} {-v:.6f}\n")
    shutil.make_archive(SUB, "zip", root_dir=SUB); shutil.rmtree(SUB)
OUT_NAMES_ = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt", ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
              ("gender", "English"): "gender/sub_score_v4_English_heard.txt", ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}

def run_experiment(name, cfg):
    cfg_apply(cfg); t0 = time.time(); ms = []
    for k in range(K_FOLDS):
        stA = stage_a(EXP_MEM, k, EXP_SEED, (EXP_A_EP,))[EXP_A_EP]
        ms.append(finetune(EXP_MEM, stA, subset(FL, np.where(FOLD != k)[0]), B_EPOCHS, EXP_SEED))
    sc = defaultdict(list); lb = defaultdict(list); raw = defaultdict(list)
    for k in range(K_FOLDS):
        idx, st, gn = OOF_TRIALS[k]; VA = subset(FL, idx); COH = subset(FL, np.where(FOLD != k)[0])
        Fx, Vx = to_t(face_in(EXP_MEM, VA)), to_t(voice_in(EXP_MEM, VA))
        for tn, tr in (("std", st), ("gender", gn)):
            sc[tn].append(member_scores(EXP_MEM, [ms[k]], VA, tr[:, 0], tr[:, 1], COH)); lb[tn].append(tr[:, 2])
            raw[tn].append(pair_scores(ms[k], Fx[tr[:, 0]], Vx[tr[:, 1]]))
    r = {f"{tn}": eer(np.concatenate(lb[tn]), np.concatenate(sc[tn])) for tn in ("std", "gender")}
    r.update({f"{tn}_raw": eer(np.concatenate(lb[tn]), np.concatenate(raw[tn])) for tn in ("std", "gender")})
    write_zip(name, [(EXP_MEM, ms)], "exp_zips")
    for k_, m in enumerate(ms):
        torch.save(m.state_dict(), f"{WORK}/exp_models/{name}_fold{k_}.pt")
    row = dict(experiment=name, **{k: (v if not isinstance(v, float) or v is None else round(v, 4)) for k, v in cfg.items()},
               nogender_like=round(r["std"], 2), gender_like=round(r["gender"], 2), avg=round((r["std"] + r["gender"]) / 2, 2),
               nogender_raw=round(r["std_raw"], 2), gender_raw=round(r["gender_raw"], 2), minutes=round((time.time() - t0) / 60, 1))
    print(f"[{name}] held-out EER  no-gender-like {r['std']:5.2f}  gender-like {r['gender']:5.2f}  (raw cosine {r['std_raw']:5.2f} / {r['gender_raw']:5.2f})"
          f"  {row['minutes']} min", flush=True)
    return row

EXP_ROWS = []
if os.environ.get("EXP_SET") == "nway" and VOX is not None:
    # v9 check: does the same-gender N-way loss help? v7's chosen configuration with/without it, 3 seeds each;
    # identical folds, Stage-A seeds and held-out trials for every variant (paired comparison).
    os.makedirs(f"{WORK}/exp_zips", exist_ok=True); os.makedirs(f"{WORK}/exp_models", exist_ok=True)
    V7 = dict(HN_W=0.5, HN_MODE="any", SAMPLING="samegender", SAME_GENDER_FRAC=0.5, TEMP=0.1, AUX_GW=0.0, FACE_ATTR=True,
              B_EPOCHS=15, B_LR=1e-4, WEIGHT_DECAY=1e-4, NWAY_W=0.0, NWAY_N=10)
    VARIANTS = [("base_v7", {}), ("nway_w0.5_n10", dict(NWAY_W=0.5)), ("nway_w1_n10", dict(NWAY_W=1.0)), ("nway_w1_n5", dict(NWAY_W=1.0, NWAY_N=5)),
                ("nway_w1_all", dict(NWAY_W=1.0, NWAY_N=0)), ("nway_w1_all_noHN", dict(NWAY_W=1.0, NWAY_N=0, HN_W=0.0))]
    for EXP_SEED in [int(x) for x in os.environ.get("EXP_SEEDS", "0,1,2").split(",")]:
        for vn, ch in VARIANTS:
            row = run_experiment(f"{vn}_seed{EXP_SEED}", dict(V7, **ch)); row.update(variant=vn, seed=EXP_SEED); EXP_ROWS.append(row)
            pd.DataFrame(EXP_ROWS).to_csv(f"{WORK}/experiments_nway.csv", index=False)
    T = pd.DataFrame(EXP_ROWS); keys = ["nogender_like", "gender_like", "avg", "nogender_raw", "gender_raw"]
    B = T[T.variant == "base_v7"].set_index("seed")[keys]
    D = pd.concat([(g.set_index("seed")[keys] - B).assign(variant=v) for v, g in T.groupby("variant", sort=False)])
    summ = T.groupby("variant", sort=False)[keys].mean().round(2)
    summ["d_gender_vs_base"] = D.groupby("variant", sort=False)["gender_like"].mean().round(2)
    summ["d_gender_sd"] = D.groupby("variant", sort=False)["gender_like"].std().round(2)
    summ["d_nogender_vs_base"] = D.groupby("variant", sort=False)["nogender_like"].mean().round(2)
    print("\nmean over seeds (held-out 7-fold OOF EER; d_ = paired difference vs base_v7, negative = better):\n" + summ.to_string(), flush=True)
    summ.to_csv(f"{WORK}/experiments_nway_summary.csv")
elif RUN_EXPERIMENTS and VOX is not None:
    os.makedirs(f"{WORK}/exp_zips", exist_ok=True); os.makedirs(f"{WORK}/exp_models", exist_ok=True)
    best_cfg = cfg_now(); base = run_experiment("E0_baseline_v6", best_cfg); EXP_ROWS.append(base); best = base
    GROUPS = [
        ("E2_hardneg", [dict(HN_MODE="off"), dict(HN_MODE="samegender")]),
        ("E3_sampling", [dict(SAMPLING="balanced"), dict(SAMPLING="samegender", SAME_GENDER_FRAC=0.5)]),
        ("E4_temperature", [dict(TEMP=0.03), dict(TEMP=0.05), dict(TEMP=0.07), dict(TEMP=0.10)]),
        ("E5_gender_shortcut", [dict(AUX_GW=0.0), dict(FACE_ATTR=False), dict(AUX_GW=0.0, FACE_ATTR=False)]),
        ("E6_finetune", [dict(B_EPOCHS=8), dict(B_EPOCHS=25), dict(B_LR=3e-4), dict(WEIGHT_DECAY=1e-3)]),
    ]
    for gname, cands in GROUPS:
        group_best = best
        for j, ch in enumerate(cands):
            cfg = dict(best_cfg, **ch); nm = f"{gname}_{j}_" + "_".join(f"{k}={v}" for k, v in ch.items())
            row = run_experiment(nm, cfg); EXP_ROWS.append(row)
            if row["gender_like"] < group_best["gender_like"] - 0.1 and row["nogender_like"] <= best["nogender_like"] + 0.5:
                group_best, group_cfg = row, cfg
        if group_best is not best:
            best, best_cfg = group_best, group_cfg
        print(f"--> after {gname}: keep {best['experiment']} (gender-like {best['gender_like']}, no-gender-like {best['nogender_like']})", flush=True)
    cfg_apply(best_cfg)
    table = pd.DataFrame(EXP_ROWS); table.to_csv(f"{WORK}/experiments.csv", index=False)
    print(table[["experiment", "nogender_like", "gender_like", "avg", "nogender_raw", "gender_raw", "minutes"]].to_string(index=False))
    print("chosen configuration:", best_cfg)
    json.dump({"chosen": best_cfg, "rows": EXP_ROWS}, open(f"{WORK}/experiments.json", "w"), indent=1, default=str)
if not RUN_EXPERIMENTS:                 # v8: reuse the configuration v7's experiments chose
    cfg_apply(json.loads(os.environ.get("CHOSEN_CFG", json.dumps(dict(HN_W=0.5, HN_MODE="any", SAMPLING="samegender", SAME_GENDER_FRAC=0.5, TEMP=0.1,
                                                                      AUX_GW=0.0, FACE_ATTR=True, B_EPOCHS=15, B_LR=1e-4, WEIGHT_DECAY=1e-4)))))
    print("configuration (from v7 experiments):", cfg_now())
for m in MEMBERS:                       # the chosen face-attribute setting applies to every member of the final ensemble
    m["face_attr"] = EXP_MEM["face_attr"]

TRAINED = {}     # name -> list over seeds of [7 fold models]
log = []; A_CHOSEN = {}
for mem in MEMBERS:
    TRAINED[mem["name"]] = []
    for seed in SEEDS:
        print(f"\n===== {mem['name']} seed {seed}  ({(time.time()-T0)/60:.1f} min)", flush=True)
        ms, es = [], []
        if VOX is None:
            dF, dV = face_in(mem, FL).shape[1], voice_in(mem, FL).shape[1]; set_seed(seed)
            init = FaceVoiceNet(dF, dV, 1, mem["hidden"], mem.get("d", EMB_DIM), mem.get("p_in", 0.0)).to(DEVICE).state_dict()
            for k in range(K_FOLDS):
                m = finetune(mem, init, subset(FL, np.where(FOLD != k)[0]), B_EPOCHS, seed); ms.append(m); es.append(oof_eer(m, mem, k))
        elif mem.get("no_stage_a"):
            for k in range(K_FOLDS):
                m = finetune_scratch(mem, subset(FL, np.where(FOLD != k)[0]), seed); ms.append(m); es.append(oof_eer(m, mem, k))
        elif mem["name"] not in A_CHOSEN:        # first seed: try every Stage-A length, keep the best by out-of-fold FLAG EER
            per = defaultdict(list); models = defaultdict(list)
            for k in range(K_FOLDS):
                for ep, st_ in stage_a(mem, k, seed, A_CKPTS).items():
                    m = finetune(mem, st_, subset(FL, np.where(FOLD != k)[0]), B_EPOCHS, seed); models[ep].append(m); per[ep].append(oof_eer(m, mem, k))
            A_CHOSEN[mem["name"]] = min(per, key=lambda e: np.mean(per[e])); ms, es = models[A_CHOSEN[mem["name"]]], per[A_CHOSEN[mem["name"]]]
            print("  7-fold OOF EER by Stage-A epochs: " + ", ".join(f"{e}: {np.mean(v):.2f}" for e, v in sorted(per.items()))
                  + f" -> {A_CHOSEN[mem['name']]}", flush=True)
        else:
            ep = A_CHOSEN[mem["name"]]
            for k in range(K_FOLDS):
                m = finetune(mem, stage_a(mem, k, seed, (ep,))[ep], subset(FL, np.where(FOLD != k)[0]), B_EPOCHS, seed)
                ms.append(m); es.append(oof_eer(m, mem, k))
        print(f"  7-fold out-of-fold EER: {np.mean(es):.2f}", flush=True)
        for k, m in enumerate(ms):
            torch.save(m.state_dict(), f"{WORK}/models/{mem['name']}_seed{seed}_fold{k}.pt")
        TRAINED[mem["name"]].append(ms)
        log.append({"member": mem["name"], "seed": seed, "A_epochs": A_CHOSEN.get(mem["name"]), "oof_eer": float(np.mean(es))})
print(pd.DataFrame(log).round(2).to_string(index=False))

# %%
# ================= ENSEMBLE WEIGHTS from out-of-fold scores (all 70 ids) =================
names = [m["name"] for m in MEMBERS]; oof = defaultdict(list); lab = defaultdict(list)
for k in range(K_FOLDS):
    idx, st, gn = OOF_TRIALS[k]; VA = subset(FL, idx); COH = subset(FL, np.where(FOLD != k)[0])
    for tn, tr in (("std", st), ("gender", gn)):
        if tr is None: continue
        lab[tn].append(tr[:, 2])
        for mem in MEMBERS:
            oof[(mem["name"], tn)].append(member_scores(mem, [ms[k] for ms in TRAINED[mem["name"]]], VA, tr[:, 0], tr[:, 1], COH))
lab = {t: np.concatenate(v) for t, v in lab.items()}; oof = {k: np.concatenate(v) for k, v in oof.items()}
for n in names:
    print(f"member {n:12s} OOF EER  random-neg {eer(lab['std'], oof[(n, 'std')]):5.2f}  same-gender {eer(lab['gender'], oof[(n, 'gender')]):5.2f}")
grid = [w for w in np.array(np.meshgrid(*[np.arange(0, 1.01, 0.25)] * len(names))).T.reshape(-1, len(names)) if abs(w.sum() - 1) < 1e-6]
ens = lambda w: np.mean([eer(lab[t], sum(wi * oof[(n, t)] for wi, n in zip(w, names))) for t in ("std", "gender")])
res = {tuple(w): ens(w) for w in grid}; W = dict(zip(names, min(res, key=res.get)))
if os.environ.get("FIXED_W"):      # v10: weights fixed in advance (the OOF tuning is biased when the fine-tuned encoders saw all 70 ids)
    W = {n: float(json.loads(os.environ["FIXED_W"]).get(n, 0.0)) for n in names}; print("FIXED ensemble weights:", W)
print("ensemble weights:", W, f"| OOF EER {min(res.values()):.2f} | equal weights {ens(np.ones(len(names)) / len(names)):.2f}")
log.append({"ensemble_weights": W, "ensemble_oof_eer": min(res.values())})

# %%
# ================= SYSTEM A (test-list clustering, no trial structure) tuned out-of-fold =================
# v11: NOT ALLOWED under the FLAG 2027 FAQ (#2) -> skipped when COMPLIANT
FACE_FEATS = {"arc": lambda S: S["arc"], "vgg": lambda S: S["vgg"], "arcvgg": lambda S: l2n(np.concatenate([S["arc"], S["vgg"]], 1))}

@torch.no_grad()
def components(fold_k, S, fi, vi, thr_f, thr_v, min_size, cohort_S, fk):
    """fold_k = k: only the models that never saw fold k (out-of-fold tuning); None: all fold models (test time)."""
    Xf = FACE_FEATS[fk](S); cf = refine(Xf, cluster(Xf, thr_f), min_size)
    cv = refine(S["ecapa"], cluster(S["ecapa"], thr_v), min_size)
    pair = np.zeros(len(fi)); Ssum = 0.0
    for mem in MEMBERS:
        w = W[mem["name"]]
        if not w:
            continue
        models = [ms[fold_k] for ms in TRAINED[mem["name"]]] if fold_k is not None else [m for ms in TRAINED[mem["name"]] for m in ms]
        pair += w * member_scores(mem, models, S, fi, vi, cohort_S)
        Fx, Vx = to_t(face_in(mem, S)), to_t(voice_in(mem, S)); Sm = []
        for m in models:
            zf, zv = project(m, Fx, "face").cpu().numpy(), project(m, Vx, "voice").cpu().numpy()
            if NORM in ("center", "both"):
                zf, zv = l2n(zf - zf.mean(0)), l2n(zv - zv.mean(0))
            CF = l2n(np.stack([zf[cf == c].mean(0) for c in range(cf.max() + 1)])); CV = l2n(np.stack([zv[cv == c].mean(0) for c in range(cv.max() + 1)]))
            Sm.append(CF @ CV.T)
        Ssum = Ssum + w * np.mean(Sm, 0)
    return pair, Ssum, cf[fi], cv[vi]

THR = {fk: None for fk in FACE_FEATS}
if COMPLIANT: FACE_FEATS = {}
for fk in [f for f in ("arcvgg", "vgg") if f in FACE_FEATS]:
    print(f"face clustering threshold ({fk}):"); THR[fk] = tune_threshold(FACE_FEATS[fk](FL), FL["spk"])
print("voice clustering threshold:"); THR_V = tune_threshold(FL["ecapa"], FL["spk"])
LISTS = []                                         # out-of-fold test-like lists: 7 folds x 2 negative types x 2 draws
for k in range(K_FOLDS):
    idx = np.where(FOLD == k)[0]; VA = subset(FL, idx); COH = subset(FL, np.where(FOLD != k)[0])
    for sg in (False, True):
        for r in range(2):
            fi, vi, lb = sim_pairs(VA["spk"], flag_gender, sg, 100 * k + 10 * r + int(sg))
            LISTS.append((k, VA, COH, fi, vi, lb))
pair_only = float("nan") if COMPLIANT else np.mean([eer(lb, components(k, VA, fi, vi, THR["vgg"], THR_V, 0, COH, "vgg")[0]) for k, VA, COH, fi, vi, lb in LISTS])
best_A = (float("inf"), None); t0 = time.time()
for fk in [f for f in ("arcvgg", "vgg") if f in FACE_FEATS]:
    for ms_ in (5, 10):
        for tf in sorted({round(max(0.2, THR[fk] + d), 2) for d in (-0.1, -0.05, 0)}):
            for tv in sorted({round(max(0.2, THR_V + d), 2) for d in (-0.1, -0.05, 0)}):
                cps = [(components(k, VA, fi, vi, tf, tv, ms_, COH, fk), lb) for k, VA, COH, fi, vi, lb in LISTS]
                for g in [(c, a, t) for c in (0, 1) for a in (1, 2, 4) for t in (0.05,)]:
                    e = np.mean([eer(lb, combine(cp, *g)) for cp, lb in cps])
                    if e < best_A[0]:
                        best_A = (e, dict(face=fk, thr_f=tf, thr_v=tv, min_size=ms_, w=dict(w_clus=g[0], w_asg=g[1], tau=g[2])))
TD_A = best_A[1]
if os.environ.get("FIXED_TDA"): TD_A = json.loads(os.environ["FIXED_TDA"]); print("FIXED System A settings:", TD_A)
print(f"out-of-fold test-like lists (7 folds x 4): pair-only {pair_only:.2f} | A {best_A[0]:.2f} {TD_A} ({(time.time()-t0)/60:.1f} min)")
log.append({"oof_pair_only": pair_only, "oof_A": best_A[0], "TD_A": TD_A})

# %%
# ================= WRITE SUBMISSIONS (A and P) =================
OUT_NAMES = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt",
             ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
             ("gender", "English"): "gender/sub_score_v4_English_heard.txt",
             ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
systems = {"P": ("submission_P_pairs_only", None), "A": ("submission_A_inputs_only", TD_A), "A_lang": ("submission_A_langnorm", TD_A)}
if COMPLIANT: systems = {"C": ("submission_compliant", None)}      # v11: each pair on its own, plain cosine, fixed weights
NORM_DEFAULT = NORM
ALL = lambda name: [m for ms in TRAINED[name] for m in ms]
for sysname, (name, TD) in systems.items():
    SUB = f"{WORK}/{name}"; shutil.rmtree(SUB, ignore_errors=True); written = []
    for path in test_paths:
        _, track, lang = os.path.basename(path)[:-4].split("__")
        S = flag_set(os.path.basename(path), False); idx = np.arange(len(S["face"]))
        NORM = ("center" if lang == "Bangla" else "base") if sysname == "A_lang" else NORM_DEFAULT      # A_lang: centring only on the unheard language
        if TD is None:      # each pair on its own: no test-set clustering (AS-norm cohort = FLAG train)
            s = sum(W[m["name"]] * member_scores(m, ALL(m["name"]), S, idx, idx, FL) for m in MEMBERS if W[m["name"]])
            if not COMPLIANT: s = zs(s)
        else:
            s = combine(components(None, S, idx, idx, TD["thr_f"], TD["thr_v"], TD["min_size"], FL, TD["face"]), **TD["w"])
        pid = np.load(path, allow_pickle=True)["pid"]
        out = f"{SUB}/{OUT_NAMES[(track, lang)]}"; os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            for p, sc in zip(pid, s):
                fh.write(f"{p} {-sc:.6f}\n")      # official scorer: LOWER score = same person
        written.append(OUT_NAMES[(track, lang)]); print(f"{sysname} {OUT_NAMES[(track, lang)]:45s} {len(s)} pairs")
    assert set(written) == set(OUT_NAMES.values()), f"missing: {set(OUT_NAMES.values()) - set(written)}"
    shutil.make_archive(f"{WORK}/{name}", "zip", root_dir=SUB); NORM = NORM_DEFAULT
json.dump({"phase": PHASE, "experiments": EXP_ROWS, "config": cfg_now(), "norm": NORM, "same_gender_frac": SAME_GENDER_FRAC, "hard_neg": HN_W, "lang_adv": LANG_ADV, "face": FACE_SRC, "attributes": ATTR_SRC, "folds": K_FOLDS, "seeds": SEEDS, "members": names, "log": log},
          open(f"{WORK}/run_summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"done in {(time.time()-T0)/60:.1f} min")
