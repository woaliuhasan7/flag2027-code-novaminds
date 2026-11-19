# FLAG 2027 — FULL cross-modal fine-tuning of the face AND voice encoders on one local GPU (RTX 4070, 12 GB).
# Architecture unchanged: face encoder (VGGFace2 InceptionResnetV1) -> tower -> L2;  voice encoder (WavLM-base-plus-SV) -> tower -> L2;
# score = cosine. ALL layers of both encoders are trained (WavLM's raw-waveform CNN front end stays frozen, as is standard);
# lower layers get smaller learning rates; BatchNorm statistics of the face encoder are kept fixed; gradient checkpointing
# keeps it inside 12 GB.
# Data (no Bengali/Bangla speech anywhere: VoxLingua107 language ID, speaker excluded if any clip looks Bengali):
#   * VoxCeleb2 (more clips per speaker when vox-more-* data is present, else the 4-clip sample)  -> face<->voice pairs
#   * FLAG train (English part only), MINUS N_HOLD identities held out for validation          -> face<->voice pairs
#   * VoxCeleb1 (audio only on our mirror)                                                       -> voice speaker loss only
# Loss: bidirectional cross-modal SupCon (T = 0.1) + AAM speaker loss on each encoder output (+ on VoxCeleb1 voices).
# v9 options: + same-gender N-way matching loss (nway_w, nway_n); query pooling over WavLM frames next to the x-vector (qpool).
# Batches: P identities x 2 clips; a fraction SAME_G of batches from ONE gender (same-gender negatives); a fraction FLAG_FRAC
# of batches from FLAG identities only. Gender only builds batches; it is never used at inference.
# Phase 0: encoders frozen, towers trained (best = "frozen baseline");  Phase 1: everything trained.
# Checkpoint selection: mean EER over (VoxCeleb2 test speakers, FLAG held-out ids) x (random negatives, same-gender negatives).
# Outputs in OUT/xft/<name>/: summary.json, progress.json, log.txt, holdout_ids.json, encoders_best.pt,
#   flag/<set>.npz (face_enc, face_joint, voice_enc, voice_joint + spk/stem/pid, same item order as gpu-extract-1),
#   vox/<tag>.npz (per crop: face_enc/face_joint + crop_clip_idx; per clip: voice_enc/voice_joint, clip, spk) in fp16.
import os, sys, glob, json, math, time, random, io, importlib.util, types
from collections import defaultdict, Counter, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, pandas as pd
from sklearn.metrics import roc_curve
from torch.utils.checkpoint import checkpoint

CFG = json.loads(os.environ.get("CFG", "{}"))
IN, OUT = os.environ["IN_ROOT"], os.environ["OUT_ROOT"]; SMOKE = bool(os.environ.get("SMOKE"))
V = CFG.get("name", "full_sg50"); O = os.path.join(OUT, "xft", V); os.makedirs(O, exist_ok=True)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"; AMP = DEV != "cpu"
SEED = CFG.get("seed", 0); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4); cv2.setNumThreads(2)
P_IDS, K_PER, PV = CFG.get("p", 64), 2, CFG.get("pv", 24)
CROP, TEMP, AAM_W = 48000, 0.1, CFG.get("aam_w", 0.3)
SAME_G, FLAG_FRAC = CFG.get("same_g", 0.5), CFG.get("flag_frac", 0.2)
STEPS0, STEPS, EVAL_EVERY = CFG.get("steps0", 2000), CFG.get("steps", 8000), CFG.get("eval_every", 500)
LR_V, LR_F, LR_H, DECAY_V = CFG.get("lr_v", 2e-5), CFG.get("lr_f", 3e-5), 1e-3, CFG.get("decay_v", 0.85)
N_HOLD, BUDGET = CFG.get("n_hold", 14), CFG.get("budget_min", 1e9)
NWAY_W, NWAY_N = CFG.get("nway_w", 0.0), CFG.get("nway_n", 10)   # v9: same-gender N-way matching loss (0 = off)
QPOOL = bool(CFG.get("qpool", False))                           # v9: learnable-query attention pooling over WavLM frames (next to the x-vector)
VD = 1024 if QPOOL else 512                                     # voice-encoder output size (x-vector 512 [+ query pooling 512])
T0 = time.time(); LOGF = open(os.path.join(O, "log.txt"), "a", encoding="utf-8")
def log(*a):
    s = f"[{V} {(time.time()-T0)/60:7.1f} min] " + " ".join(map(str, a)); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def l2n(x): x = np.asarray(x, np.float32); return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)
def eer(l, s):
    if len(set(np.asarray(l).tolist())) < 2: return float("nan")
    f, t, _ = roc_curve(l, s); n = 1 - t; i = np.nanargmin(np.abs(n - f)); return 100 * (f[i] + n[i]) / 2
def dec(b): return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
def fit(x, L=CROP, rand=True):
    if len(x) >= L:
        s = random.randint(0, len(x) - L) if rand else 0; return x[s:s + L]
    return np.tile(x, int(np.ceil(L / max(len(x), 1))))[:L]
log("config:", json.dumps(dict(CFG, device=DEV)))

# ======================================================================= data sources
class VoxNew:                                   # vox-more-*: audio.bin (FLAC back to back) + meta.npz (+ crops.npz)
    def __init__(s, d):
        m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True); s.tag = os.path.basename(d); n = len(m["clip"])
        s.clip, s.spk, s.off = m["clip"], m["spk"], m["a_off"]; s.lang, s.pbn = m["lang"], m["p_bn"]
        s.bin = np.memmap(os.path.join(d, "audio.bin"), np.uint8, "r")
        s.good = [[] for _ in range(n)]; s.allc = [[] for _ in range(n)]; s.male = [[] for _ in range(n)]; s.jpg = None
        cp = os.path.join(d, "crops.npz")
        if os.path.exists(cp):
            c = np.load(cp, allow_pickle=True); s.jpg = c["jpg"]
            for r, (ci, st, ml) in enumerate(zip(c["clip_idx"], c["status"], c["male"])):
                if ci < n:
                    s.allc[ci].append(r)
                    if str(st) != "nodet": s.good[ci].append(r)
                    if ml >= 0: s.male[ci].append(ml)
        s.extra = {k: m[k] for k in ("g_spk", "g_male") if k in m.files}
    def __len__(s): return len(s.clip)
    def audio(s, i, L=None, rand=True):
        with sf.SoundFile(io.BytesIO(s.bin[s.off[i]:s.off[i + 1]].tobytes())) as f:
            n = f.frames
            if L and n > L:
                f.seek(random.randint(0, n - L) if rand else 0); return f.read(L, dtype="float32")
            return f.read(dtype="float32")
class VoxOld:                                   # 4-clip sample: gpu-extract-* (audio_int16 + meta) + crop-cache-* (crops)
    def __init__(s, meta_p, crops_p):
        m = np.load(meta_p, allow_pickle=True); c = np.load(crops_p, allow_pickle=True); s.tag = os.path.basename(os.path.dirname(meta_p)); n = len(m["clip"])
        s.clip, s.spk, s.off = m["clip"], m["spk"], m["offsets"]; s.lang = np.array(["?"] * n); s.pbn = np.zeros(n)
        s.A = np.load(os.path.join(os.path.dirname(meta_p), "audio_int16.npy"), mmap_mode="r"); s.jpg = c["jpg"]
        rows = defaultdict(list)
        for r, cc in enumerate(c["clip"]): rows[str(cc)].append(r)
        s.allc = [rows.get(str(cl), []) for cl in s.clip]; s.good = [[r for r in rr if str(c["status"][r]) != "nodet"] for rr in s.allc]
        s.male = [[] for _ in range(n)]
        for j, ci in enumerate(m["face_clip"]):
            if m["male"][j] >= 0: s.male[int(ci)].append(m["male"][j])
        s.extra = {}
    def __len__(s): return len(s.clip)
    def audio(s, i, L=None, rand=True):
        x = np.asarray(s.A[s.off[i]:s.off[i + 1]], np.float32) / 32768.0
        if L and len(x) > L:
            st = random.randint(0, len(x) - L) if rand else 0; x = x[st:st + L]
        return x

def biggest(pattern):                            # one copy per name: the most complete one
    best = {}
    for p in glob.glob(os.path.join(IN, pattern), recursive=True):
        k = os.path.basename(os.path.dirname(p)); n = os.path.getsize(p)
        if k not in best or n > best[k][1]: best[k] = (p, n)
    return {k: v[0] for k, v in sorted(best.items())}
NEW = {k: os.path.dirname(p) for k, p in biggest("**/m_vox*/meta.npz").items()}
OLDM, OLDC = biggest("**/vox/vox2_*/meta.npz"), biggest("**/vox_crops/vox2_*/crops.npz")
SRC = {}
for t in sorted(set(OLDM) | {k[2:] for k in NEW if k.startswith("m_vox2")}):
    if f"m_{t}" in NEW: SRC[t] = VoxNew(NEW[f"m_{t}"])
    elif t in OLDM and t in OLDC: SRC[t] = VoxOld(OLDM[t], OLDC[t])
VOX1 = {k: VoxNew(d) for k, d in NEW.items() if k.startswith("m_vox1")}
for t, s in SRC.items(): log(f"VoxCeleb2 {t}: {'MORE clips (vox-more)' if isinstance(s, VoxNew) else '4-clip sample'} | {len(s)} clips, {len(set(s.spk))} speakers")
for t, s in VOX1.items(): log(f"VoxCeleb1 {t}: {len(s)} clips, {len(set(s.spk))} speakers (audio only)")

# Bengali exclusion: earlier language-ID list + the per-clip language ID of the new data
EXCL = set()
for p in glob.glob(os.path.join(IN, "**/langid/bengali_speakers.json"), recursive=True): EXCL |= {f"v2:{x}" for x in json.load(open(p))["bengali_speakers"]}
def lang_excl(src, pre):
    by = defaultdict(list)
    for sp, l, pb in zip(src.spk, src.lang, src.pbn):
        if str(l) != "?": by[f"{pre}{sp}"].append((str(l), float(pb)))
    return {s for s, v in by.items() if max(pb for _, pb in v) >= 0.5 or Counter(l for l, _ in v).most_common(1)[0][0].lower().startswith("bn")}
for s in SRC.values(): EXCL |= lang_excl(s, "v2:")
for s in VOX1.values(): EXCL |= lang_excl(s, "v1:")
log("speakers excluded as Bengali:", len(EXCL))

# ---------------- FLAG (train English + test lists)
FLAG_D = {}
for p in sorted(glob.glob(os.path.join(IN, "**/flag/*.npz"), recursive=True)): FLAG_D.setdefault(os.path.basename(p), p)
def flag_crops(name, n):
    cands = sorted(glob.glob(os.path.join(IN, "**/flag_crops", name), recursive=True), key=lambda q: ("crop-cache-3" not in q, q))
    for q in cands:
        c = np.load(q, allow_pickle=True)
        if len(c["jpg"]) == n: return c["jpg"]
    raise FileNotFoundError(f"flag_crops/{name} with {n} items")
def load_wav(p, max_s=12):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(1)[: int(max_s * sr)]
    if sr != 16000:
        import torchaudio; x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    return x
FLAG_G = {}
mcsv = glob.glob(os.path.join(IN, "**/flag/*meta*.csv"), recursive=True)
if mcsv:
    df = pd.read_csv(mcsv[0], sep=None, engine="python", header=None, dtype=str).apply(lambda c: c.str.strip().str.lower())
    idc = next(c for c in df.columns if df[c].str.match(r"^id\d+$").mean() > .6); gc = next(c for c in df.columns if df[c].isin(["m", "f", "male", "female"]).mean() > .6)
    FLAG_G = {i: float(g[0] == "m") for i, g in zip(df[idc], df[gc])}
ft = np.load(FLAG_D["train.npz"], allow_pickle=True); FT_JPG = flag_crops("train.npz", len(ft["spk"])); FSPK = np.array([str(s) for s in ft["spk"]])
ids = sorted(set(FSPK)); nh = min(N_HOLD, len(ids) // 3)            # n_hold 0 = final run on ALL FLAG train ids
if CFG.get("holdout"): HOLD = sorted(CFG["holdout"])
else:
    rs = np.random.default_rng(2027); HOLD = []
    for g in (1.0, 0.0):                                                  # stratified by gender
        pool = [i for i in ids if FLAG_G.get(i) == g]; k = int(round(nh * len(pool) / len(ids)))
        HOLD += list(rs.choice(pool, min(k, len(pool)), replace=False))
    HOLD = sorted(map(str, HOLD))
json.dump({"holdout_ids": HOLD, "n_ids": len(ids)}, open(os.path.join(O, "holdout_ids.json"), "w"), indent=1)
F_TR = [j for j in range(len(FSPK)) if FSPK[j] not in HOLD and FSPK[j] in FLAG_G]
F_VA = [j for j in range(len(FSPK)) if FSPK[j] in HOLD]
FAUD = {}
for j in F_TR + F_VA:
    w = str(ft["wav"][j])
    if w not in FAUD: FAUD[w] = (np.clip(load_wav(os.path.join(IN, w), 12), -1, 1) * 32767).astype(np.int16)
log(f"FLAG train: {len(ids)} ids, {len(F_TR)} items used for training; held out for validation: {HOLD} ({len(F_VA)} items)")

# ---------------- training table: (source, index, speaker label)
SPK_G, ENT = {}, []                               # gender per speaker key; entries (kind, src_key, i, spk_key)
g_of = defaultdict(list)
for t, s in SRC.items():
    for i in range(len(s)):
        if s.male[i]: g_of[f"v2:{s.spk[i]}"] += list(s.male[i])
VOX_G = {k: float(np.mean(v) >= .5) for k, v in g_of.items()}
TEST_TAG = "vox2_test_mp4"
for t, s in SRC.items():
    if t == TEST_TAG: continue
    for i in range(len(s)):
        k = f"v2:{s.spk[i]}"
        if s.good[i] and k not in EXCL and k in VOX_G: ENT.append(("v2", t, i, k)); SPK_G[k] = VOX_G[k]
for j in F_TR:
    k = f"fl:{FSPK[j]}"; ENT.append(("fl", "flag", j, k)); SPK_G[k] = FLAG_G[FSPK[j]]
LAB = {k: n for n, k in enumerate(sorted(SPK_G))}
BY = defaultdict(list)
for e, (_, _, _, k) in enumerate(ENT): BY[k].append(e)
V2_SPK = [k for k in BY if k.startswith("v2:")]; FL_SPK = [k for k in BY if k.startswith("fl:")]
V1E, V1BY = [], defaultdict(list)
for t, s in VOX1.items():
    for i in range(len(s)):
        k = f"v1:{s.spk[i]}"
        if k not in EXCL: V1BY[k].append(len(V1E)); V1E.append((t, i))
V1LAB = {k: n for n, k in enumerate(sorted(V1BY))}
VA = [i for i in range(len(SRC[TEST_TAG])) if f"v2:{SRC[TEST_TAG].spk[i]}" not in EXCL and SRC[TEST_TAG].allc[i]] if TEST_TAG in SRC else []
log(f"cross-modal training: {len(ENT)} clips | VoxCeleb2 {len(V2_SPK)} speakers ({Counter(SPK_G[k] for k in V2_SPK)}) | FLAG {len(FL_SPK)} ids"
    f" | VoxCeleb1 voice-only: {len(V1E)} clips, {len(V1BY)} speakers | VoxCeleb2 test (validation): {len(VA)} clips")
if SMOKE and not VA: raise SystemExit("smoke data needs a vox2_test_mp4 part")

def make_batch(item):
    ents, v1 = item; F_, W_, Y = [], [], []
    for e in ents:
        kind, t, i, k = ENT[e]
        if kind == "v2":
            s = SRC[t]; f = dec(s.jpg[random.choice(s.good[i])]); w = fit(s.audio(i, CROP))
        else:
            f = dec(FT_JPG[i]); w = fit(FAUD[str(ft["wav"][i])].astype(np.float32) / 32767.0)
        if random.random() < .5: f = f[:, ::-1]
        F_.append(np.ascontiguousarray(f)); W_.append(w.astype(np.float32)); Y.append(LAB[k])
    W1, Y1 = [], []
    for e in v1:
        t, i = V1E[e]; W1.append(fit(VOX1[t].audio(i, CROP)).astype(np.float32)); Y1.append(V1LAB[f"v1:{VOX1[t].spk[i]}"])
    return (torch.from_numpy(np.stack(F_)), torch.from_numpy(np.stack(W_ + W1)), torch.tensor(Y), torch.tensor(Y1, dtype=torch.long))
def sampler(steps, seed, use_v1):
    rng = random.Random(seed)
    for _ in range(steps):
        pool = FL_SPK if (FL_SPK and rng.random() < FLAG_FRAC) else V2_SPK
        if rng.random() < SAME_G:
            g = 1.0 if rng.random() < .5 else 0.0; pool = [k for k in pool if SPK_G[k] == g] or pool
        ss = rng.sample(pool, min(P_IDS, len(pool)))
        ents = [e for k in ss for e in (rng.sample(BY[k], K_PER) if len(BY[k]) >= K_PER else rng.choices(BY[k], k=K_PER))]
        v1 = []
        if use_v1 and V1BY:
            for k in rng.sample(list(V1BY), min(PV, len(V1BY))): v1 += rng.sample(V1BY[k], 2) if len(V1BY[k]) >= 2 else rng.choices(V1BY[k], k=2)
        yield ents, v1
def prefetch(gen, n=4, ahead=10):
    with ThreadPoolExecutor(n) as ex:
        q = deque()
        for it in gen:
            q.append(ex.submit(make_batch, it))
            if len(q) >= ahead: yield q.popleft().result()
        while q: yield q.popleft().result()

# ======================================================================= encoders + towers
def load_irv1():
    root = importlib.util.find_spec("facenet_pytorch").submodule_search_locations[0]      # skip facenet_pytorch/__init__ (MTCNN/torchvision)
    for nm, pth in (("facenet_pytorch", root), ("facenet_pytorch.models", os.path.join(root, "models")), ("facenet_pytorch.models.utils", os.path.join(root, "models", "utils"))):
        if nm not in sys.modules:
            mod = types.ModuleType(nm); mod.__path__ = [pth]; sys.modules[nm] = mod
    from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
    return InceptionResnetV1(pretrained=None if SMOKE else "vggface2")
face = load_irv1().to(DEV); face.dropout.p = 0.2
if SMOKE:
    from transformers import WavLMConfig, WavLMForXVector
    wl = WavLMForXVector(WavLMConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=2, intermediate_size=128, conv_dim=(32,) * 7,
                                     tdnn_dim=(32, 32, 32, 32, 64), xvector_output_dim=512, use_weighted_layer_sum=True)).to(DEV)
else:
    from transformers import WavLMForXVector
    wl = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEV)
wl.config.layerdrop = 0.0; wl.freeze_feature_encoder()
wl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
for p in list(face.parameters()) + list(wl.parameters()): p.requires_grad = False
FACE_GROUPS = [(0.1, [face.conv2d_1a, face.conv2d_2a, face.conv2d_2b, face.conv2d_3b, face.conv2d_4a, face.conv2d_4b]),
               (0.3, [face.repeat_1, face.mixed_6a]), (0.6, [face.repeat_2, face.mixed_7a]),
               (1.0, [face.repeat_3, face.block8, face.last_linear, face.last_bn])]
nL = len(wl.wavlm.encoder.layers)
VOICE_GROUPS = [(DECAY_V ** nL, [wl.wavlm.feature_projection, wl.wavlm.encoder.pos_conv_embed, wl.wavlm.encoder.layer_norm])]
VOICE_GROUPS += [(DECAY_V ** (nL - 1 - i), [wl.wavlm.encoder.layers[i]]) for i in range(nL)]
VOICE_GROUPS += [(1.0, [wl.projector, wl.tdnn, wl.feature_extractor])]
VOICE_EXTRA = [wl.layer_weights] if getattr(wl, "layer_weights", None) is not None else []
ENC_PARAMS = [p for _, ms in FACE_GROUPS + VOICE_GROUPS for m in ms for p in m.parameters()] + VOICE_EXTRA
class QueryPool(nn.Module):
    """v9 query pooling: its own softmax weights over all WavLM hidden layers, then M learnable queries attend over the frames
    (multi-head attention) -> one 512-d voice vector. Sits NEXT TO the pretrained x-vector head (that head is unchanged)."""
    def __init__(s, n_layers, d=768, dk=256, m=4, heads=4, out=512):
        super().__init__(); s.lw = nn.Parameter(torch.zeros(n_layers)); s.ln = nn.LayerNorm(d); s.kv = nn.Linear(d, dk)
        s.q = nn.Parameter(torch.randn(m, dk) * 0.02); s.att = nn.MultiheadAttention(dk, heads, batch_first=True)
        s.out = nn.Sequential(nn.LayerNorm(m * dk), nn.Linear(m * dk, out))
    def forward(s, hs):                                           # hs: (B, layers, frames, d)
        x = s.kv(s.ln((hs * F.softmax(s.lw.float(), 0).to(hs.dtype).view(1, -1, 1, 1)).sum(1)))
        o, _ = s.att(s.q.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1), x, x)
        return s.out(o.flatten(1))
QP = QueryPool(wl.config.num_hidden_layers + 1, wl.config.hidden_size).to(DEV) if QPOOL else None
class Tower(nn.Module):
    def __init__(s, din=512, d=256, h=512, p=0.3, p_in=0.2):
        super().__init__(); s.drop = nn.Dropout(p_in)
        s.net = nn.Sequential(nn.Linear(din, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, d)); s.skip = nn.Linear(din, d)
    def forward(s, x): x = s.drop(x); return F.normalize(s.net(x) + s.skip(x), dim=-1)
face_t, voice_t = Tower().to(DEV), Tower(din=VD).to(DEV)
Wf = nn.Parameter(torch.randn(len(LAB), 512, device=DEV) * .01); Wv = nn.Parameter(torch.randn(len(LAB), VD, device=DEV) * .01)
W1 = nn.Parameter(torch.randn(max(len(V1LAB), 1), VD, device=DEV) * .01)
HEADS = list(face_t.parameters()) + list(voice_t.parameters()) + [Wf, Wv, W1] + (list(QP.parameters()) if QP is not None else [])
LAB_G = torch.tensor([SPK_G[k] for k in sorted(SPK_G)], device=DEV)    # gender per training label: ONLY to pick N-way distractors
log(f"v9 options: same-gender N-way loss weight {NWAY_W} (N = {NWAY_N if NWAY_N > 0 else 'all'}) | query pooling {'ON' if QPOOL else 'off'}"
    + (f" ({sum(p.numel() for p in QP.parameters())/1e6:.2f} M params)" if QP is not None else ""))
log(f"encoder params trained in phase 1: face {sum(p.numel() for _, ms in FACE_GROUPS for m in ms for p in m.parameters())/1e6:.1f} M, "
    f"voice {sum(p.numel() for _, ms in VOICE_GROUPS for m in ms for p in m.parameters())/1e6:.1f} M")

def set_mode(train, enc_train):
    face.eval(); wl.eval(); face_t.train(train); voice_t.train(train)
    if QP is not None: QP.train(train)
    if train and enc_train:
        face.train(); wl.train()
        for m in face.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d): m.eval()      # keep pretrained BN statistics
def face_fwd(x_uint8):
    x = (x_uint8.to(DEV).permute(0, 3, 1, 2).float() - 127.5) / 128.0; m = face
    ck = (lambda f, a: checkpoint(f, a, use_reentrant=False)) if (m.training and torch.is_grad_enabled()) else (lambda f, a: f(a))
    def stem(x):
        for l in (m.conv2d_1a, m.conv2d_2a, m.conv2d_2b, m.maxpool_3a, m.conv2d_3b, m.conv2d_4a, m.conv2d_4b): x = l(x)
        return x
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        x = ck(stem, x)
        for b in list(m.repeat_1) + [m.mixed_6a] + list(m.repeat_2) + [m.mixed_7a] + list(m.repeat_3): x = ck(b, x)
        x = m.block8(x); x = m.avgpool_1a(x); x = m.dropout(x); x = m.last_bn(m.last_linear(x.view(x.shape[0], -1)))
    return F.normalize(x.float(), dim=-1)
def voice_fwd(w, enc_grad=False):
    """WavLM-SV x-vector embedding (same computation as WavLMForXVector.forward), plus - with query pooling - the query-pooled
    vector from the same forward pass. enc_grad: gradients through WavLM itself (phase 1); the query-pooling head always learns."""
    w = (w - w.mean(1, keepdim=True)) / torch.sqrt(w.var(1, keepdim=True) + 1e-7)
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        with torch.set_grad_enabled(enc_grad and torch.is_grad_enabled()):
            o = wl.wavlm(w.to(DEV), output_hidden_states=True, return_dict=True); hs = torch.stack(o.hidden_states, 1)
            if wl.config.use_weighted_layer_sum:
                h = (hs * F.softmax(wl.layer_weights, dim=-1).view(-1, 1, 1)).sum(1)
            else:
                h = o.last_hidden_state
            h = wl.projector(h)
            for t in wl.tdnn: h = t(h)
            e = F.normalize(wl.feature_extractor(torch.cat([h.mean(1), h.std(1)], -1)).float(), dim=-1)
        if QP is None: return e
        q = F.normalize(QP(hs).float(), dim=-1)
    return F.normalize(torch.cat([e, q], -1), dim=-1)
def supcon(zf, zv, y):
    lg = zf @ zv.T / TEMP; pos = (y[:, None] == y[None, :]).float()
    side = lambda l: -((l - torch.logsumexp(l, 1, keepdim=True)) * pos).sum(1).div(pos.sum(1)).mean()
    return .5 * (side(lg) + side(lg.T))
def nway_sg(zf, zv, y, g, n):
    """v9 same-gender N-way matching: every voice must pick a face of its own identity out of its own-identity faces + (n-1)
    random faces of OTHER identities of the SAME gender (and every face the matching voice). n <= 0: all same-gender faces in the
    batch. Gender only chooses the distractors during training; it never enters the embeddings or the score."""
    s = zf @ zv.T / TEMP; pos = y[:, None] == y[None, :]; neg = (~pos) & (g[:, None] == g[None, :]) & (g[:, None] >= 0)
    def side(l, pos, neg):
        if n > 0:
            r = torch.rand_like(l).masked_fill(~neg, -1.0); neg = neg & (r >= r.topk(min(n - 1, l.shape[1]), dim=1).values[:, -1:])
        ok = neg.any(1)
        if not ok.any(): return l.new_zeros(())
        l = l.masked_fill(~(pos | neg), float("-inf")); lp = l - torch.logsumexp(l, 1, keepdim=True); p = pos.float()
        return -(lp.masked_fill(~pos, 0.0) * p).sum(1).div(p.sum(1))[ok].mean()
    return .5 * (side(s, pos, neg) + side(s.T, pos.T, neg.T))
def aam(z, W, y, m=0.2, s=30.0):
    cos = z @ F.normalize(W, dim=-1).T; th = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
    return F.cross_entropy(s * torch.where(F.one_hot(y, W.shape[0]).bool(), torch.cos(th + m), cos), y)

# ======================================================================= evaluation
@torch.no_grad()
def emb_faces(jpgs, bs=256):
    set_mode(False, False); E, J = [], []
    for i in range(0, len(jpgs), bs):
        e = face_fwd(torch.from_numpy(np.stack([dec(b) for b in jpgs[i:i + bs]]))); E.append(e.cpu()); J.append(face_t(e).cpu())
    return (torch.cat(E).numpy(), torch.cat(J).numpy()) if E else (np.zeros((0, 512), np.float32), np.zeros((0, 256), np.float32))
@torch.no_grad()
def emb_voices(waves, bs=32):
    set_mode(False, False); waves = [w if len(w) >= 16000 else np.tile(w, int(np.ceil(16000 / max(len(w), 1))))[:16000] for w in waves]
    E, J = np.zeros((len(waves), VD), np.float32), np.zeros((len(waves), 256), np.float32); by = defaultdict(list)
    for i, w in enumerate(waves): by[len(w)].append(i)
    for L, idx in by.items():                                     # equal lengths -> batched, no padding
        for k in range(0, len(idx), bs):
            ii = idx[k:k + bs]; e = voice_fwd(torch.from_numpy(np.stack([np.asarray(waves[i], np.float32) for i in ii])))
            E[ii] = e.cpu().numpy(); J[ii] = voice_t(e).cpu().numpy()
    return E, J
def trials_eer(zf, zv, spk, g_of):
    rng = np.random.default_rng(11); by = defaultdict(list)
    for j, s in enumerate(spk): by[s].append(j)
    out = {}
    for name, same in (("std", False), ("gender", True)):
        a, b, l = [], [], []
        for j, s in enumerate(spk):
            o = [k for k in by[s] if k != j]
            if not o: continue
            a.append(j); b.append(rng.choice(o)); l.append(1)
            neg = [t for t in by if t != s and (not same or (g_of.get(t) is not None and g_of.get(t) == g_of.get(s)))]
            if neg: a.append(j); b.append(rng.choice(by[rng.choice(neg)])); l.append(0)
        out[name] = eer(np.array(l), (zf[a] * zv[b]).sum(1))
    return out
VT = SRC.get(TEST_TAG)
VA_AUD = [VT.audio(i, 6 * 16000, rand=False) for i in VA]
FV_AUD = [FAUD[str(ft["wav"][j])][: 8 * 16000].astype(np.float32) / 32767.0 for j in F_VA]
def evaluate():
    out = {}
    if VA:
        rows = [(i, r) for i in VA for r in (VT.good[i] or VT.allc[i])]
        _, fj = emb_faces([VT.jpg[r] for _, r in rows]); acc = defaultdict(list)
        for (i, _), z in zip(rows, fj): acc[i].append(z)
        _, vj = emb_voices(VA_AUD)
        r = trials_eer(np.stack([l2n(np.mean(acc[i], 0)) for i in VA]), vj, [str(VT.spk[i]) for i in VA], {str(k[3:]): g for k, g in VOX_G.items()})
        out.update(vox_std=r["std"], vox_gender=r["gender"])
    _, f2 = emb_faces([FT_JPG[j] for j in F_VA]); _, v2 = emb_voices(FV_AUD)
    r = trials_eer(f2, v2, [FSPK[j] for j in F_VA], FLAG_G); out.update(flag_std=r["std"], flag_gender=r["gender"])
    out["score"] = float(np.nanmean([v for v in out.values()]))
    return {k: float(v) for k, v in out.items()}

# ======================================================================= training (resumable)
# Everything needed to continue is saved to disk: ckpt_last.pt (models + optimizer + schedule + step, every eval in phase 1 and at
# the end of phase 0) and encoders_best.pt (best checkpoint so far). Running the same command again continues where it stopped;
# if training had finished, only the missing feature files are written. CFG "fresh": start over.
LAST_P, BEST_P, SUMM_P = (os.path.join(O, f) for f in ("ckpt_last.pt", "encoders_best.pt", "summary.json"))
if CFG.get("fresh"):
    for f_ in (LAST_P, BEST_P, SUMM_P):
        if os.path.exists(f_): os.remove(f_)
    log("fresh start: previous checkpoints of this run removed")
def atomic_save(obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)
def load_pt(path): return torch.load(path, map_location="cpu", weights_only=False)
HIST = []
def snapshot():
    cpu = lambda m: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return {"face": cpu(face), "wl": cpu(wl), "ft": cpu(face_t), "vt": cpu(voice_t), **({"qp": cpu(QP)} if QP is not None else {})}
def heads_w(): return {"Wf": Wf.detach().cpu().clone(), "Wv": Wv.detach().cpu().clone(), "W1": W1.detach().cpu().clone()}
def restore(models, W=None):
    face.load_state_dict(models["face"]); wl.load_state_dict(models["wl"]); face_t.load_state_dict(models["ft"]); voice_t.load_state_dict(models["vt"])
    if QP is not None: QP.load_state_dict(models["qp"])
    if W is not None:
        with torch.no_grad(): Wf.copy_(W["Wf"]); Wv.copy_(W["Wv"]); W1.copy_(W["W1"])
def run(phase, steps, enc_train, best, resume=None):
    groups = [{"params": HEADS, "lr": LR_H * (0.3 if enc_train else 1.0), "base": LR_H * (0.3 if enc_train else 1.0)}]
    if enc_train:
        for p in ENC_PARAMS: p.requires_grad = True
        groups += [{"params": [p for m in ms for p in m.parameters()], "lr": LR_F * f, "base": LR_F * f} for f, ms in FACE_GROUPS]
        groups += [{"params": [p for m in ms for p in m.parameters()], "lr": LR_V * f, "base": LR_V * f} for f, ms in VOICE_GROUPS]
        if VOICE_EXTRA: groups.append({"params": VOICE_EXTRA, "lr": LR_V, "base": LR_V})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    warm = max(1, min(300, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP); losses = []; t1 = time.time(); start = 0
    if resume is not None:
        opt.load_state_dict(resume["opt"]); sched.load_state_dict(resume["sched"]); scaler.load_state_dict(resume["scaler"]); start = resume["step"]
        resume.clear()                                                            # free the RAM copy
        log(f"[{phase}] RESUMED at step {start}/{steps} (best so far: {best[2]})")
    params = HEADS + (ENC_PARAMS if enc_train else [])
    gen = sampler(steps, SEED + (7 if enc_train else 0), enc_train)
    for _ in range(start): next(gen)                                             # same batch order as an uninterrupted run
    for step, (f, w, y, y1) in enumerate(prefetch(gen), start=start):
        set_mode(True, enc_train); y, y1 = y.to(DEV), y1.to(DEV); B = len(y)
        opt.zero_grad(set_to_none=True); l1 = 0.0
        if enc_train and len(y1):           # VoxCeleb1 voices (speaker loss only): own forward + backward FIRST, so their activations are
            L1 = AAM_W * aam(voice_fwd(w[B:], enc_train), W1, y1)        # freed before the main batch -> same gradients, ~1.5 GB less GPU memory
            scaler.scale(L1).backward(); l1 = float(L1); del L1
        with torch.set_grad_enabled(enc_train):
            ef = face_fwd(f)
        ev = voice_fwd(w[:B], enc_train); zf_, zv_ = face_t(ef), voice_t(ev)
        L = supcon(zf_, zv_, y) + AAM_W * (aam(ef, Wf, y) + aam(ev, Wv, y))
        if NWAY_W: L = L + NWAY_W * nway_sg(zf_, zv_, y, LAB_G[y], NWAY_N)
        scaler.scale(L).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 5.0); scaler.step(opt); scaler.update(); sched.step(); losses.append(L.item() + l1)
        if (step + 1) % 50 == 0:
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if AMP else 0; res = torch.cuda.max_memory_reserved() / 2 ** 30 if AMP else 0
            free = torch.cuda.mem_get_info()[0] / 2 ** 30 if AMP else 0
            log(f"[{phase}] step {step+1}/{steps} loss {np.mean(losses[-50:]):.3f} | {(time.time()-t1)/(step+1-start):.2f} s/step | GPU peak {mem:.1f} GB"
                f" (reserved {res:.1f}, card free now {free:.1f})")
        stop = (time.time() - T0) / 60 > BUDGET
        take = enc_train and step + 1 == CFG.get("take_step", -1)                # final run: keep exactly this step of the run-1 schedule
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == steps or stop or take:
            r = evaluate(); HIST.append(dict(phase=phase, step=step + 1, loss=float(np.mean(losses[-EVAL_EVERY:])), **r))
            log(f"[{phase}] step {step+1} EVAL " + " ".join(f"{k} {v:.2f}" for k, v in r.items()))
            if (r["score"] < best[0] and not (enc_train and CFG.get("take_step"))) or take:
                best = (r["score"], None, dict(HIST[-1]))
                if enc_train: atomic_save(dict(snapshot(), cfg=CFG, baseline=BASE, best=best[2], holdout=HOLD, partial=True), BEST_P)
            json.dump({"cfg": CFG, "history": HIST}, open(os.path.join(O, "progress.json"), "w"), indent=1)
            if enc_train and step + 1 < steps and not (stop or take):
                atomic_save({"phase": 1, "step": step + 1, "models": snapshot(), "W": heads_w(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                             "scaler": scaler.state_dict(), "best": best[:1] + (None,) + best[2:], "hist": HIST, "base": BASE, "cfg": CFG}, LAST_P)
                log(f"[{phase}] checkpoint saved at step {step+1} (a restart continues from here)")
                if CFG.get("die_at") == step + 1: os._exit(3)                       # test hook: simulate a crash right after a checkpoint
        if stop: log(f"time budget reached at step {step+1}"); break
        if take: log(f"final run: checkpoint taken at step {step+1}"); break
    for p in ENC_PARAMS: p.requires_grad = False
    return best

if sys.platform == "win32":                       # keep the PC from going to sleep while this runs (no admin rights needed)
    try:
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception: pass
BASE = None
if (CFG.get("only_extract") or os.path.exists(SUMM_P)) and os.path.exists(BEST_P):
    st = load_pt(BEST_P); restore(st); BASE, best = st.get("baseline"), (0, None, st.get("best"))
    log("training of this run had already finished -> loading its best checkpoint, writing the feature files that are missing")
else:
    ck = load_pt(LAST_P) if os.path.exists(LAST_P) else None
    if ck is not None and ck.get("cfg", {}).get("name") != CFG.get("name"): ck = None
    if ck is None:
        BASE = run("0 frozen encoders", STEPS0, False, (1e9, None, None))[2]; log("frozen-encoder baseline (best phase-0 eval):", BASE)
        atomic_save({"phase": 0, "step": 0, "models": snapshot(), "W": heads_w(), "hist": HIST, "base": BASE, "cfg": CFG}, LAST_P)
        best = run("1 full fine-tuning", STEPS, True, (1e9, None, None))
    else:
        restore(ck["models"], ck["W"]); HIST[:] = ck["hist"]; BASE = ck["base"]
        log(f"found a saved checkpoint: phase {ck['phase']} step {ck['step']} -> continuing (frozen-encoder baseline: {BASE})")
        if ck["phase"] == 0:
            best = run("1 full fine-tuning", STEPS, True, (1e9, None, None))
        else:
            b = ck["best"]; best = run("1 full fine-tuning", STEPS, True, (b[0], None, b[2]), resume=ck)
        del ck
    log("best fine-tuned checkpoint:", best[2])
    if best[2] is None or not os.path.exists(BEST_P): raise SystemExit("no fine-tuned checkpoint was evaluated")
    st = load_pt(BEST_P); restore(st); atomic_save(dict(st, cfg=CFG, baseline=BASE, best=best[2], holdout=HOLD, partial=False), BEST_P); del st
    json.dump({"cfg": CFG, "frozen_baseline": BASE, "best": best[2], "history": HIST, "holdout_ids": HOLD, "excluded_bengali": len(EXCL)},
              open(SUMM_P, "w"), indent=1)
    for f_ in (LAST_P, LAST_P + ".tmp", BEST_P + ".tmp"):                    # training finished: the 1.5 GB resume file is no longer needed
        if os.path.exists(f_): os.remove(f_)

if CFG.get("no_extract"): log("FINISHED (speed test, no feature extraction)"); sys.exit(0)
# ======================================================================= features with the selected encoders (files already written are kept)
os.makedirs(os.path.join(O, "flag"), exist_ok=True); os.makedirs(os.path.join(O, "vox"), exist_ok=True)
def done_npz(path, key):
    try:
        z = np.load(path, allow_pickle=True); return key in z.files and len(z[key]) > 0
    except Exception: return False
def save_npz(path, **kw):
    tmp = path[:-4] + ".tmp.npz"; np.savez(tmp, **kw); os.replace(tmp, path)
for name, p in FLAG_D.items():
    outp = os.path.join(O, "flag", name if name.endswith(".npz") else name + ".npz")
    if done_npz(outp, "voice_joint"): log(f"features FLAG {name}: already written"); continue
    d = np.load(p, allow_pickle=True); n = len(d["wav"]); fe, fj = emb_faces(list(flag_crops(name, n)))
    uniq = sorted(set(map(str, d["wav"]))); pos = {w: k for k, w in enumerate(uniq)}
    ve, vj = emb_voices([load_wav(os.path.join(IN, w), 12) for w in uniq]); sel = [pos[str(w)] for w in d["wav"]]
    extra = {k: d[k] for k in ("spk", "stem", "pid") if k in d.files}
    save_npz(outp, face_enc=fe, face_joint=fj, voice_enc=ve[sel], voice_joint=vj[sel], **extra)
    log(f"features FLAG {name}: {n} items, {len(uniq)} unique voices")
h = lambda a: np.asarray(a, np.float16)
for t, s in ({} if CFG.get("no_vox_feats") else SRC).items():        # no_vox_feats: seed runs only need the FLAG feature files
    outp = os.path.join(O, "vox", f"{t}.npz")
    if done_npz(outp, "voice_joint"): log(f"features VoxCeleb2 {t}: already written"); continue
    rows = [(i, r) for i in range(len(s)) for r in s.allc[i]]
    fe, fj = emb_faces([s.jpg[r] for _, r in rows])
    ve, vj = np.zeros((0, VD)), np.zeros((0, 256))
    for k in range(0, len(s), 2000):                                   # chunks keep RAM low
        a, b = emb_voices([s.audio(i, 6 * 16000, rand=False) for i in range(k, min(k + 2000, len(s)))]); ve = np.concatenate([ve, a]); vj = np.concatenate([vj, b])
    save_npz(outp, crop_clip_idx=np.array([i for i, _ in rows], np.int32), crop_row=np.array([r for _, r in rows], np.int32),
             clip=s.clip, spk=s.spk, face_enc=h(fe), face_joint=h(fj), voice_enc=h(ve), voice_joint=h(vj), source=np.array("more" if isinstance(s, VoxNew) else "sample"))
    log(f"features VoxCeleb2 {t}: {len(rows)} crops, {len(s)} clips")
log("FINISHED")
