"""FLAG 2027 Option A - FINAL models on all 70 English FLAG training identities + official evaluation scoring (Kaggle CPU).
Same production training code (nb2v11.py, COMPLIANT=1) and the same recipe as the 7-fold CV runs (adv_cv.py):
  Stage A (VoxCeleb2, Bengali speakers removed, VoxCeleb2 fold 0 held out for monitoring only) 12 epochs; Stage B (all 70 FLAG ids)
  15 epochs; fixed epochs, no checkpoint selection. Seeds 0, 1, 2. Stage-B seed = seed*1000 + 17*7 (predeclared; the CV folds use 17*k, k = 0..6).
  M1 = VGGFace2 + ReDimNet2-B3 (production B3 features, dataset flag-rdn-b3); M2 = VGGFace2 + CAM++ (official stats->dense 512-d;
  train / VoxCeleb2 features from adv-x0a..x3, evaluation-list features from optA-evalx). Face input = VGGFace2 512 + ViT age.
Evaluation scoring: each trial on its own, plain cosine of the two tower outputs per seed (no normalisation, no cohort, no test statistics).
Outputs (per model): towers per seed, manifests, sha256, raw per-seed cosines for the 4 official lists, run log."""
import glob, hashlib, json, os, platform, shutil, sys, time
import numpy as np
MODEL = "M4ADA"   # AdaFace IR-101 (VGGFace2 slot) + CAM++, production tower recipe                 # M1 | M2
FINAL_SEEDS = [0, 1, 2]                                              # (name differs from nb2v11.SEEDS, which the exec overwrites)
T0 = time.time(); OUT = f"/kaggle/working/{MODEL}"; WORK = "/kaggle/tmp/work"; IN = "/kaggle/tmp/in"
for _d in (WORK, IN, "/kaggle/tmp/voice"): shutil.rmtree(_d, ignore_errors=True)
os.makedirs(f"{WORK}/models", exist_ok=True); os.makedirs(IN, exist_ok=True); os.makedirs(OUT, exist_ok=True)
def one(pat):
    h = sorted(glob.glob(pat, recursive=True)); assert h, pat; return h[0]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
V11 = os.path.dirname(os.path.dirname(os.path.dirname(one("/kaggle/input/**/gpu-extract-1/flag/train.npz"))))
CODE = os.path.dirname(one("/kaggle/input/**/adv_code_marker.txt")); NB = f"{CODE}/nb2v11.py"
RB3 = os.path.dirname(os.path.dirname(one("/kaggle/input/**/flag-rdn-b3/**/vox/vox2_mp4_1.npz")))
META = {"model": MODEL, "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "python": platform.python_version(), "nb2v11_sha256": sha(NB)}
for d in os.listdir(V11):
    if os.path.isdir(f"{V11}/{d}") and not d.startswith("final_all70") and d not in ("rdn", "vgg1", "vgg2"):
        os.symlink(f"{V11}/{d}", f"{IN}/{d}")
tr0 = np.load(f"{V11}/gpu-extract-1/flag/train.npz", allow_pickle=True)
if MODEL == "M1":
    os.symlink(RB3, f"{IN}/rdn"); META["voice"] = f"ReDimNet2-B3 production features {RB3}"
else:
    R = "/kaggle/tmp/voice/rdn"; os.makedirs(f"{R}/flag", exist_ok=True); os.makedirs(f"{R}/vox", exist_ok=True)
    f = np.load(one("/kaggle/input/**/adv/flag/train_feats.npz"), allow_pickle=True)
    assert (f["stem"].astype(str) == tr0["stem"].astype(str)).all() and (f["spk"].astype(str) == tr0["spk"].astype(str)).all()
    np.savez(f"{R}/flag/train.npz", rdn2=f["campp_official"].astype(np.float32), spk=f["spk"], stem=f["stem"])
    for tagf in sorted(glob.glob(f"{RB3}/vox/*.npz")):
        tag = os.path.basename(tagf)[:-4]; ref = np.load(tagf, allow_pickle=True); z = np.load(one(f"/kaggle/input/**/adv/vox/{tag}.npz"), allow_pickle=True)
        assert (z["meta_clip"].astype(str) == ref["meta_clip"].astype(str)).all(), tag
        np.savez(f"{R}/vox/{tag}.npz", meta_clip=z["meta_clip"], rdn2=z["campp_official"].astype(np.float32))
    for p in sorted(glob.glob("/kaggle/input/**/adv/flag_eval/eval__*.npz", recursive=True)):
        e = np.load(p, allow_pickle=True); ref = np.load(f"{V11}/gpu-extract-1/flag/{os.path.basename(p)}", allow_pickle=True)
        assert (e["pid"].astype(str) == ref["pid"].astype(str)).all(), p
        np.savez(f"{R}/flag/{os.path.basename(p)}", rdn2=e["campp_official"].astype(np.float32), pid=e["pid"])
    assert len(glob.glob(f"{R}/flag/eval__*.npz")) == 4
    os.symlink(R, f"{IN}/rdn"); META["voice"] = "CAM++ official (adv-x0a..x3 train / VoxCeleb2; optA-evalx evaluation lists), key campp_official"
A = os.path.dirname(os.path.dirname(one("/kaggle/input/**/flag-ada-vgg/**/flag_vgg/train.npz"))); AD = "/kaggle/tmp/ada"
os.makedirs(f"{AD}/flag_vgg", exist_ok=True); os.symlink(f"{A}/vox_vgg", f"{AD}/vox_vgg"); os.symlink(f"{A}/flag_vgg/train.npz", f"{AD}/flag_vgg/train.npz")
_ev = sorted(glob.glob("/kaggle/input/**/flag-ada-eval/**/eval__*.npz", recursive=True)); assert len(_ev) == 4, _ev
for p in _ev:
    e = np.load(p, allow_pickle=True); ref = np.load(f"{V11}/vgg1/flag_vgg/{os.path.basename(p)}", allow_pickle=True)
    assert (e["pid"].astype(str) == ref["pid"].astype(str)).all(), p
    os.symlink(p, f"{AD}/flag_vgg/{os.path.basename(p)}")
os.symlink(AD, f"{IN}/ada_as_vgg"); META["face"] = "AdaFace IR-101 WebFace12M in the VGGFace2 slot (train/VoxCeleb2: flag-ada-vgg; evaluation lists: flag-ada-eval)"
print("input farm:", sorted(os.listdir(IN)), flush=True)
os.environ.update(FLAG_INPUT_ROOT=IN, FLAG_WORK=WORK, FACE_SRC="vgg", ATTR_SRC="vit", SEEDS="0,1", COMPLIANT="1", MEMBERS="ew_aux,rdn_aux", PHASE="eval")
src = open(NB).read()
def part(a, b=None):
    s = src.split(a, 1)[1]; return a + (s.split(b, 1)[0] if b else s)
exec(compile(src.split("# ================= 7-FOLD SPLITS")[0], "head", "exec"))
exec(compile(part("# ================= 7-FOLD SPLITS", "# %%\n# ================= TRAIN"), "splits", "exec"))
exec(compile(part("# ================= TRAIN: members", "def run_experiment"), "defs", "exec"))
torch.set_num_threads(2); META["torch"] = torch.__version__; META["threads"] = 2
BASE = dict(HN_W=0.5, HN_MODE="any", SAMPLING="samegender", SAME_GENDER_FRAC=0.5, TEMP=0.1, AUX_GW=0.0, FACE_ATTR=True,
            B_EPOCHS=15, B_LR=1e-4, WEIGHT_DECAY=1e-4, P_IDS=64, K_PER_ID=4)
for m in MEMBERS: m["face_attr"] = "age"
cfg_apply({**BASE, "B_STEP_SCALE": 1.0, "DEMO_A_BANDS": 0, "CLIP_DIVERSE": False, "VBANK_P": 0.0, "QUEUE_A": 0, "PROTO_A_W": 0.0})
assert COMPLIANT and not USE_AS_NORM and NORM == "base", (COMPLIANT, USE_AS_NORM, NORM)
mem = [m for m in MEMBERS if m["name"] == "rdn_aux"][0]
ids = sorted(set(FL["spk"])); assert len(ids) == 70, len(ids)
META["recipe"] = {**BASE, "stage_a_epochs": 12, "stage_a_vox_val_fold": 0, "stage_b_seed": "seed*1000 + 17*7", "member": mem}
META["data"] = {"flag_train_items": int(len(FL["spk"])), "flag_ids": len(ids), "vox_speakers": int(len(set(VOX["spk"]))), "vox_frames": int(len(VOX["face"])),
                "face_dim": int(face_in(mem, FL).shape[1]), "voice_dim": int(voice_in(mem, FL).shape[1])}
print("FINAL", MODEL, META["data"], flush=True)
tests = sorted(test_paths); assert len(tests) == 4 and all(os.path.basename(p).startswith("eval__") for p in tests), tests
EV = {}
for p in tests:
    n = os.path.basename(p); S = flag_set(n, False); EV[n] = (S, np.load(p, allow_pickle=True)["pid"].astype(str))
SC = {}
for seed in FINAL_SEEDS:
    t0 = time.time(); stA = stage_a(mem, 0, seed, (12,))[12]; tA = time.time() - t0
    set_seed(seed * 1000 + 17 * 7); m = finetune(mem, stA, FL, B_EPOCHS, seed); m.eval()
    d = f"{OUT}/seed_{seed}"; os.makedirs(d, exist_ok=True)
    torch.save(m.state_dict(), f"{d}/model_full.pt"); torch.save(m.face.state_dict(), f"{d}/face_tower.pt"); torch.save(m.voice.state_dict(), f"{d}/voice_tower.pt")
    manifest = {"model": MODEL, "seed": seed, "training_identities": ids, "n_identities": len(ids), "n_items": int(len(FL["spk"])),
                "items_per_identity": {s: int((FL["spk"] == s).sum()) for s in ids}, "all_70_used": len(ids) == 70,
                "bengali_used": False, "evaluation_items_used_for_training": False}
    json.dump(manifest, open(f"{d}/training_manifest.json", "w"), indent=1)
    for n, (S, pid) in EV.items():
        idx = np.arange(len(S["face"])); SC[f"{n[:-4]}__seed{seed}"] = member_scores(mem, [m], S, idx, idx, None).astype(np.float64)
    cfg = {"model": MODEL, "seed": seed, "stage_b_seed": seed * 1000 + 17 * 7, "minutes": (time.time() - t0) / 60, "stage_a_minutes": tA / 60,
           "sha256": {k: sha(f"{d}/{k}") for k in ("model_full.pt", "face_tower.pt", "voice_tower.pt", "training_manifest.json")}}
    json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
    print(f"[final] {MODEL} seed {seed}: {(time.time() - t0) / 60:.1f} min (Stage A {tA / 60:.1f}) | sha face {cfg['sha256']['face_tower.pt'][:12]} voice {cfg['sha256']['voice_tower.pt'][:12]}", flush=True)
np.savez(f"{OUT}/eval_cosines_{MODEL}.npz", **SC, **{f"{n[:-4]}__pid": pid for n, (S, pid) in EV.items()})
# per-trial independence check: re-score the first 50 trials of every list one at a time; must equal the batched scores
mx = 0.0
for n, (S, pid) in EV.items():
    ms = [m]
    for i in range(50):
        one_ = {k: (v[i:i + 1] if isinstance(v, np.ndarray) and len(v) == len(S["face"]) else v) for k, v in S.items()}
        mx = max(mx, abs(float(member_scores(mem, ms, one_, np.array([0]), np.array([0]), None)[0]) - SC[f"{n[:-4]}__seed2"][i]))
META["single_vs_batched_max_abs_diff_seed2"] = mx; assert mx < 1e-5, mx
META["end_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()); META["minutes"] = (time.time() - T0) / 60
META["eval_lists"] = {n: len(pid) for n, (S, pid) in EV.items()}
json.dump(META, open(f"{OUT}/run_meta_{MODEL}.json", "w"), indent=1, default=str)
shutil.rmtree("/kaggle/tmp", ignore_errors=True); print("done", META["eval_lists"], "single-vs-batched", mx, flush=True)
