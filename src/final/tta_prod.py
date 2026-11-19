"""FLAG 2027 - production 20.33 ensemble + horizontal-flip face TTA (variant T1s of the TTA study), inference only, Kaggle CPU.
Towers = the exact production all-70 towers (final_<member>_seed{0,1,2}.pt), weights = production (rdn/vb2/clip 0.25, xj0 runs 1/12 each).
TTA: the VGGFace2 face embedding of each evaluation item becomes l2n(stored + l2n(VGGFace2(mirrored face crop))) for the members whose
face input is VGGFace2 (rdn_aux, vb2_aux, clip_aux); CLIP features and the fine-tuned xj0 joint embeddings are unchanged.
Each trial is scored on its own (plain cosine, mean over tower seeds; no normalisation, no cross-trial statistics). Lower = same person.
Step 1 recomputes the production scores WITHOUT TTA and requires them to match the 20.33 files (rounding level) before writing anything."""
import os, sys, glob, json, shutil, zipfile, hashlib
HERE = os.path.dirname(glob.glob("/kaggle/input/**/flag-tta-in/tta_prod.py", recursive=True)[0])
IN = "/kaggle/working/in"; os.makedirs(IN, exist_ok=True)
v11 = [p for p in glob.glob("/kaggle/input/**/flag-v11-in", recursive=True) if os.path.isdir(p)][0]
extra = [p for p in glob.glob("/kaggle/input/**/flag-ho-extra", recursive=True) if os.path.isdir(p)][0]
for d in os.listdir(v11):
    if os.path.isdir(f"{v11}/{d}"): os.symlink(f"{v11}/{d}", f"{IN}/{d}")
if not os.path.exists(f"{IN}/rdn"): os.symlink(f"{extra}/rdn", f"{IN}/rdn")
print("inputs:", sorted(os.listdir(IN)), flush=True)
CFGJ = json.load(open(f"{HERE}/final_config_ageonly_rdn.json")); W = CFGJ["weights"]; print("production weights:", W, flush=True)
os.environ.update(FLAG_INPUT_ROOT=IN, FLAG_WORK="/kaggle/working/work", FACE_SRC="vgg", ATTR_SRC="vit", SEEDS="0,1", COMPLIANT="1", PHASE="eval",
                  MEMBERS=",".join(["ew_aux"] + list(W)))
os.makedirs("/kaggle/working/work", exist_ok=True)
src = open(f"{HERE}/nb2v11.py").read()
def part(a, b=None):
    s = src.split(a, 1)[1]; return a + (s.split(b, 1)[0] if b else s)
exec(compile(src.split("# ================= 7-FOLD SPLITS")[0], "head", "exec"))
exec(compile(part("# ================= 7-FOLD SPLITS", "# %%\n# ================= TRAIN"), "splits", "exec"))
exec(compile(part("# ================= TRAIN: members", "def run_experiment"), "defs", "exec"))
torch.set_num_threads(4)
BASE = dict(HN_W=0.5, HN_MODE="any", SAMPLING="samegender", SAME_GENDER_FRAC=0.5, TEMP=0.1, AUX_GW=0.0, FACE_ATTR=True,
            B_EPOCHS=15, B_LR=1e-4, WEIGHT_DECAY=1e-4, P_IDS=64, K_PER_ID=4)
cfg_apply({**BASE, "B_STEP_SCALE": 1.0, "DEMO_A_BANDS": 0, "CLIP_DIVERSE": False, "VBANK_P": 0.0, "QUEUE_A": 0, "PROTO_A_W": 0.0})
for m in MEMBERS: m["face_attr"] = "age"
MEM = {m["name"]: m for m in MEMBERS}; assert set(W) <= set(MEM), (set(W) - set(MEM))
def load_models(mem, S):
    out = []
    for seed in (0, 1, 2):
        sd = torch.load(f"{HERE}/models/final_{mem['name']}_seed{seed}.pt", map_location="cpu")
        dF, dV = face_in(mem, S).shape[1], voice_in(mem, S).shape[1]
        m = FaceVoiceNet(dF, dV, sd["protos"].shape[0], mem["hidden"], mem.get("d", EMB_DIM), mem.get("p_in", 0.0)).to(DEVICE)
        m.load_state_dict(sd); m.eval(); out.append(m)
    return out
FILES = {("no_gender", "English"): "no_gender/sub_score_v4_English_heard.txt", ("no_gender", "Bangla"): "no_gender/sub_score_v4_Bangla_unheard.txt",
         ("gender", "English"): "gender/sub_score_v4_English_heard.txt", ("gender", "Bangla"): "gender/sub_score_v4_Bangla_unheard.txt"}
PZ = glob.glob(f"{HERE}/**/production_20p33.zip", recursive=True)
def prod_lines(rel):
    if PZ: return zipfile.ZipFile(PZ[0]).read(rel).decode().strip().split("\n")
    f = glob.glob(f"{HERE}/**/{rel}", recursive=True); assert len(f) == 1, (rel, f); return open(f[0]).read().strip().split("\n")
TTA_MEMBERS = [n for n in W if "face" in MEM[n].get("face", ("face",)) and not n.startswith("xj")]
print("TTA applied to the VGGFace2 face input of:", TTA_MEMBERS, flush=True)
SUB = "/kaggle/working/submission_final_20p33_fliptta"; shutil.rmtree(SUB, ignore_errors=True); REP = {}
MODELS = {}
for (track, lang), rel in FILES.items():
    name = f"eval__{track}__{lang}.npz"; S = flag_set(name, False); n = len(S["face"]); idx = np.arange(n)
    pid = np.load(f"{FLAG_DIR}/{name}", allow_pickle=True)["pid"].astype(str)
    fz = np.load(f"{HERE}/flipfeat/{name}", allow_pickle=True); assert (fz["pid"].astype(str) == pid).all() and len(fz["vgg_flip"]) == n
    ST = dict(S); ST["face"] = l2n(S["face"] + l2n(fz["vgg_flip"].astype(np.float32)))          # T1s: stored embedding + mirrored-crop embedding
    for k in W:
        if k not in MODELS: MODELS[k] = load_models(MEM[k], S)
    PER = {k: member_scores(MEM[k], MODELS[k], S, idx, idx, None) for k in W}
    np.savez(f"/kaggle/working/members_{name[:-4]}.npz", pid=pid, **PER)
    s0 = sum(W[k] * PER[k] for k in W)
    s1 = sum(W[k] * member_scores(MEM[k], MODELS[k], ST if k in TTA_MEMBERS else S, idx, idx, None) for k in W)
    prod = [l.split() for l in prod_lines(rel)]; v = np.array([float(x[1]) for x in prod])
    assert [x[0] for x in prod] == pid.tolist() and len(prod) == n
    rep_err = float(np.abs(-s0 - v).max()); print(f"{rel}: reproduction of 20.33 without TTA max|diff| = {rep_err:.2e}", flush=True)
    assert rep_err < 1e-4, "production scores not reproduced - stopping"
    assert np.isfinite(s1).all()
    out = f"{SUB}/{rel}"; os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        for p_, x in zip(pid, s1): fh.write(f"{p_} {-x:.6f}\n")
    back = [l.split() for l in open(out)]; assert [b[0] for b in back] == pid.tolist() and len(back) == n
    r_ = lambda a: np.argsort(np.argsort(a))
    REP[rel] = dict(rows=n, reproduction_max_abs=rep_err, spearman_vs_production=float(np.corrcoef(r_(v), r_(-s1))[0, 1]),
                    mean_abs_change=float(np.abs(-s1 - v).mean()), mean_cos_stored_vs_flip=float((l2n(S["face"]) * l2n(fz["vgg_flip"])).sum(1).mean()))
assert set(REP) == set(FILES.values())
zp = SUB + ".zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ("gender/", "no_gender/"): z.writestr(d, "")
    for rel in FILES.values(): z.write(f"{SUB}/{rel}", rel)
z = zipfile.ZipFile(zp); assert sorted(x for x in z.namelist() if x.endswith(".txt")) == sorted(FILES.values())
REP["sha256"] = hashlib.sha256(open(zp, "rb").read()).hexdigest(); REP["weights"] = W; REP["tta_members"] = TTA_MEMBERS
json.dump(REP, open(SUB + "_report.json", "w"), indent=1); print(json.dumps(REP, indent=1)); print("VALIDATED", zp)
