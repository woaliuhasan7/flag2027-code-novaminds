# FLAG 2027 advanced experiments 2 and 3 - frozen voice features (CPU, inference only, deterministic).
# Experiment 2 (temporal aggregation): for every clip, ONE forward pass of the frozen encoder; the frame-level tensor just before
#   the official pooling is pooled in several ways (no encoder weight is changed):
#     ReDimNet2-B3 (PalabraAI/redimnet2, b3, lm, VoxCeleb2-dev):
#       official = ASTP (pretrained attentive statistics) -> BN -> Linear -> 192 (exactly model(wav); checked)
#       astp     = the pretrained ASTP output (attention-weighted mean + std) BEFORE the BN/Linear embedding layer
#       mean     = uniform temporal mean of the frame features
#       meanstd  = uniform temporal mean + std of the frame features
#     CAM++ (3D-Speaker iic/speech_campplus_sv_en_voxceleb_16k v1.0.2):
#       official = statistics pooling (mean + std) -> DenseLayer (Conv1d 1x1 + BN) -> 512 (exactly model(fbank); checked)
#       meanstd  = the official statistics vector BEFORE the dense layer
#       mean     = uniform temporal mean of the frame features
# Experiment 3 (duration): FLAG English train clips only, official pooling, deterministic CENTRED crops of 2, 3 and 5 s taken from
#   the whole decoded clip (clips not longer than the target are used whole); "full" = the production policy (first 20 s, tile < 1 s).
#   Also per-clip signal measurements used ONLY for evaluation strata: natural duration, an energy-percentile noise proxy and a
#   spectral-rolloff bandwidth proxy (estimates, not labels).
# Items: FLAG English train (same order as gpu-extract-1 train.npz) and the Stage-A VoxCeleb2 clips (gpu-extract-1/2/3 meta, audio
#   copied unchanged to mahiwoaliuhasan/flag-voxaud-1/2/3). Clip policy identical to the production extraction:
#   mono 16 kHz, whole clip capped at 20 s, clips < 1 s tiled to 1 s.
import glob, hashlib, json, os, resource, subprocess, sys, time
JOBS = (os.environ.get("JOBS") or "__JOBS__").split(",")
SAMPLE = int(os.environ.get("SAMPLE") or "__SAMPLE__")            # >0: only the first n items of every job (pipeline test)
T0 = time.time()
def log(*a): print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print(c[:110], "->", r.returncode, (r.stderr or "")[-400:] if r.returncode else "", flush=True)
    return (r.stdout or "").strip()
sh(f"{sys.executable} -m pip install -q modelscope addict")
sh("git clone -q --depth 1 https://github.com/modelscope/3D-Speaker /kaggle/tmp/3D-Speaker")
sh("git clone -q --depth 1 https://github.com/PalabraAI/redimnet2 /kaggle/tmp/redimnet2")
COMMITS = {r: sh(f"git -C /kaggle/tmp/{r} rev-parse HEAD") for r in ("3D-Speaker", "redimnet2")}
import numpy as np, torch, soundfile as sf, torchaudio
import torch.nn.functional as F
torch.set_num_threads(4)
IN, OUT = "/kaggle/input", "/kaggle/working/adv"
os.makedirs(f"{OUT}/flag", exist_ok=True); os.makedirs(f"{OUT}/vox", exist_ok=True)
REP = {"jobs": JOBS, "sample": SAMPLE, "commits": COMMITS, "torch": torch.__version__, "threads": torch.get_num_threads(), "timing": {}, "checks": {}, "shapes": {}}

# ---------------------------------------------------------------- CAM++ (official speakerlab code and checkpoint)
sys.path.insert(0, "/kaggle/tmp/3D-Speaker")
import speakerlab.bin.infer_sv as inf
from speakerlab.utils.builder import dynamic_import
from speakerlab.process.processor import FBank
from speakerlab.models.campplus.layers import statistics_pooling
from modelscope.hub.snapshot_download import snapshot_download
keys = [k for k in inf.supports if "campplus_sv_en_voxceleb_16k" in k]; assert len(keys) == 1, keys
MID = keys[0]; SUP = inf.supports[MID]
for _try in range(12):                                                           # ModelScope rate-limits parallel downloads
    try:
        cache = snapshot_download(MID, revision=SUP.get("revision"), cache_dir="/kaggle/tmp/ms"); break
    except Exception as ex:
        log("ModelScope download failed, retrying:", repr(ex)[:160]); time.sleep(30 + 30 * _try)
else:
    raise SystemExit("CAM++ checkpoint download failed")
pt = os.path.join(cache, SUP["model_pt"]); sha = hashlib.sha256(open(pt, "rb").read()).hexdigest()
assert sha.startswith("5b1a88b6f8d85826"), sha                                     # same file as the M2 / M4 extraction
CAM = dynamic_import(SUP["model"]["obj"])(**SUP["model"]["args"]); CAM.load_state_dict(torch.load(pt, map_location="cpu")); CAM.eval()
FB = FBank(80, sample_rate=16000, mean_nor=True)
REP["campp"] = {"model_id": MID, "revision": SUP.get("revision"), "sha256": sha, "args": SUP["model"]["args"],
                "params": sum(p.numel() for p in CAM.parameters()), "fbank": "FBank(80, 16000, mean_nor=True)"}
log("CAM++", REP["campp"])

# ---------------------------------------------------------------- ReDimNet2-B3 (official code; release v1.0.0 weights)
RDN = torch.hub.load("/kaggle/tmp/redimnet2", "redimnet2", source="local", model_name="b3", train_type="lm", pretrained=True).eval()
REP["rdn"] = {"repo": "PalabraAI/redimnet2", "model": "b3", "train_type": "lm", "dataset": "vox2",
              "weights": "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/b3-vox2-lm.pt",
              "params": sum(p.numel() for p in RDN.parameters()), "pooling": type(RDN.pool).__name__,
              "global_context_att": bool(getattr(RDN.pool, "global_context_att", False)), "pool_in_dim": int(RDN.pool.in_dim),
              "pool_out_dim": int(RDN.pool_out_dim), "embed_dim": int(RDN.embed_dim), "emb_bn": RDN.bn2 is not None,
              "before_pool_offset": RDN.before_pool_offset, "pad_right_samples": RDN.pad_right_samples, "spec": type(RDN.spec).__name__}
log("ReDimNet2-B3", REP["rdn"])

MAXS = 20 * 16000
def prep(x):
    x = np.asarray(x, np.float32)[:MAXS]
    if len(x) < 16000: x = np.tile(x, int(np.ceil(16000 / max(len(x), 1))))[:16000]
    return x

@torch.no_grad()
def rdn_all(x, full=True):
    w0 = torch.from_numpy(prep(x))[None]; w = w0
    if RDN.pad_right_samples is not None: w = F.pad(w0, (0, RDN.pad_right_samples), mode="constant")
    s = RDN.spec(w)
    if s.ndim == 3: s = s.unsqueeze(1)
    H = RDN.backbone(s)
    if isinstance(H, (tuple, list)): H = H[0]
    shp4 = tuple(H.shape)
    if H.ndim == 4: b, C, Fq, T = H.shape; H = H.reshape(b, C * Fq, T)
    if RDN.before_pool_offset is not None: H = H[:, :, RDN.before_pool_offset:]
    p = RDN.pool(H); e = RDN.linear(RDN.bn(p))
    if RDN.bn2 is not None: e = RDN.bn2(e)
    out = {"official": e[0]}
    if full:
        mu, sd = H.mean(-1), H.std(-1)
        out.update(astp=p[0], mean=mu[0], meanstd=torch.cat([mu, sd], -1)[0])
    return {k: v.float().numpy() for k, v in out.items()}, (tuple(s.shape), shp4, tuple(H.shape)), w0

@torch.no_grad()
def cam_all(x, full=True):
    feat = FB(torch.from_numpy(prep(x))[None]).unsqueeze(0)                  # (1, frames, 80), as in the M2 extraction
    h = CAM.head(feat.permute(0, 2, 1))
    for name, mod in CAM.xvector.named_children():
        if name == "stats": break
        h = mod(h)
    st = statistics_pooling(h); e = CAM.xvector.dense(st)
    out = {"official": e[0]}
    if full: out.update(meanstd=st[0], mean=h.mean(-1)[0])
    return {k: v.float().numpy() for k, v in out.items()}, (tuple(feat.shape), tuple(h.shape)), feat

NCHK = {"rdn": 0, "campp": 0}
def check_official(enc, out, ref_in):
    """The decomposed forward must reproduce the official model output (first 5 clips of every job)."""
    if NCHK[enc] >= 5: return
    with torch.no_grad():
        ref = (RDN(ref_in) if enc == "rdn" else CAM(ref_in))[0].float().numpy()
    d = float(np.abs(ref - out["official"]).max()); REP["checks"].setdefault(f"{enc}_official_maxdiff", []).append(d); NCHK[enc] += 1
    assert d < 1e-4, (enc, d)

def read_wav(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(1)
    if sr != 16000: x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    return x

def rec_measures(x):
    """Signal-derived strata for Experiment 3 (estimates, not labels): natural duration, frame-energy P90-P10 (dB) as a noise /
    dynamic-range proxy, active-speech level (P90 energy, dBFS) and the 95% spectral roll-off (Hz) of the louder half of the frames."""
    n = len(x); dur = n / 16000.0
    if n < 400: return dict(dur_s=dur, energy_p90_p10_db=np.nan, level_p90_dbfs=np.nan, rolloff95_hz=np.nan)
    fr = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-10); p10, p90 = np.percentile(e, [10, 90])
    loud = fr[e >= np.median(e)] * np.hanning(400)
    P = (np.abs(np.fft.rfft(loud, 512, axis=1)) ** 2).mean(0); c = np.cumsum(P) / max(P.sum(), 1e-12)
    roll = float(np.searchsorted(c, 0.95) * 16000 / 512)
    return dict(dur_s=dur, energy_p90_p10_db=float(p90 - p10), level_p90_dbfs=float(p90), rolloff95_hz=roll)

def peak_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def resolve(w):
    p = f"{IN}/{w}"
    if os.path.exists(p): return p
    tail = w.split("flag-v4/", 1)[1]; hits = glob.glob(f"{IN}/**/flag-v4/{tail}", recursive=True)
    assert hits, f"missing audio {w}"; return hits[0]

V11 = [p for p in glob.glob(f"{IN}/**/gpu-extract-1/flag/train.npz", recursive=True)]
assert V11, "flag-v11-in not attached"

# ---------------------------------------------------------------- FLAG English train (Exp 2 variants + Exp 3 durations)
if "flag" in JOBS or "flagdur" in JOBS:
    DO_VAR, DO_DUR = "flag" in JOBS, "flagdur" in JOBS
    d = np.load(V11[0], allow_pickle=True); NI = SAMPLE or len(d["wav"]); wavs = [str(w) for w in d["wav"][:NI]]; U = sorted(set(wavs))
    log(f"FLAG train: {len(wavs)} items, {len(U)} distinct wav files")
    feats = {w: {} for w in U}; REC = {}; tm = {"rdn": 0.0, "campp": 0.0, "rdn_dur": 0.0, "campp_dur": 0.0, "audio_s": 0.0, "dur_audio_s": 0.0}
    for i, w in enumerate(U):
        x = read_wav(resolve(w)); REC[w] = rec_measures(x); tm["audio_s"] += len(prep(x)) / 16000
        if DO_VAR:
            t = time.time(); o, shp, ri = rdn_all(x); tm["rdn"] += time.time() - t; check_official("rdn", o, ri)
            feats[w].update({f"rdn_{k}": v for k, v in o.items()}); feats[w]["rdn_T"] = shp[2][-1]
            t = time.time(); o, shp2, ri = cam_all(x); tm["campp"] += time.time() - t; check_official("campp", o, ri)
            feats[w].update({f"campp_{k}": v for k, v in o.items()}); feats[w]["campp_T"] = shp2[1][-1]
            if i == 0: REP["shapes"]["flag_first_clip"] = {"seconds": len(prep(x)) / 16000, "rdn_spec_backbone_frames": shp, "campp_fbank_frames": shp2}
        for L in ((2, 3, 5) if DO_DUR else ()):
            n = L * 16000; c = x[(len(x) - n) // 2:(len(x) - n) // 2 + n] if len(x) > n else x
            tm["dur_audio_s"] += len(prep(c)) / 16000
            t = time.time(); feats[w][f"rdn_dur{L}"] = rdn_all(c, full=False)[0]["official"]; tm["rdn_dur"] += time.time() - t
            t = time.time(); feats[w][f"campp_dur{L}"] = cam_all(c, full=False)[0]["official"]; tm["campp_dur"] += time.time() - t
            feats[w][f"cropped{L}"] = len(x) > n
        if (i + 1) % 500 == 0: log(f"FLAG {i+1}/{len(U)} | rdn {tm['rdn']:.0f}s campp {tm['campp']:.0f}s | peak {peak_mb():.0f} MB")
    keys = [k for k in feats[U[0]]]
    np.savez(f"{OUT}/flag/{'train_feats' if DO_VAR else 'train_dur'}.npz", spk=d["spk"][:NI], stem=d["stem"][:NI], wav=np.array(wavs),
             **{k: np.stack([np.asarray(feats[w][k]) for w in wavs]) for k in keys})
    np.savez(f"{OUT}/flag/recmeta.npz", wav=np.array(wavs), **{k: np.array([REC[w][k] for w in wavs]) for k in REC[U[0]]})
    REP["timing"]["flag" if DO_VAR else "flagdur"] = {**tm, "n_wav": len(U), "peak_rss_mb": peak_mb()}
    REP["dims"] = {k: int(np.asarray(feats[U[0]][k]).size) for k in keys if not k.endswith("_T") and not k.startswith("cropped")}
    log("FLAG done", REP["timing"], REP["dims"])

# ---------------------------------------------------------------- VoxCeleb2 Stage-A clips (Exp 2 variants)
for tag in [j for j in JOBS if j.startswith("vox2_")]:
    mp = [p for p in glob.glob(f"{IN}/**/vox/{tag}/meta.npz", recursive=True)]; assert len(mp) >= 1, tag
    au = glob.glob(f"{IN}/**/{tag}__audio_int16.npy", recursive=True); assert len(au) == 1, (tag, au)
    m = np.load(mp[0], allow_pickle=True); A = np.load(au[0], mmap_mode="r"); off = m["offsets"]; n = min(len(m["clip"]), SAMPLE or len(m["clip"]))
    assert off[-1] == len(A), (tag, off[-1], len(A))
    R = {}; tm = {"rdn": 0.0, "campp": 0.0, "audio_s": 0.0}
    for i in range(n):
        x = np.asarray(A[off[i]:off[i + 1]], np.float32) / 32768.0; tm["audio_s"] += len(prep(x)) / 16000
        t = time.time(); o, shp, ri = rdn_all(x); tm["rdn"] += time.time() - t; check_official("rdn", o, ri)
        for k, v in o.items(): R.setdefault(f"rdn_{k}", []).append(v)
        R.setdefault("rdn_T", []).append(shp[2][-1])
        t = time.time(); o, shp2, ri = cam_all(x); tm["campp"] += time.time() - t; check_official("campp", o, ri)
        for k, v in o.items(): R.setdefault(f"campp_{k}", []).append(v)
        R.setdefault("campp_T", []).append(shp2[1][-1])
        if (i + 1) % 1000 == 0: log(f"{tag} {i+1}/{n} | rdn {tm['rdn']:.0f}s campp {tm['campp']:.0f}s | peak {peak_mb():.0f} MB")
    np.savez(f"{OUT}/vox/{tag}.npz", meta_clip=m["clip"][:n], **{k: np.stack(v).astype(np.float32 if not k.endswith("_T") else np.int32) for k, v in R.items()})
    REP["timing"][tag] = {**tm, "n": n, "peak_rss_mb": peak_mb()}; log(f"{tag} done", REP["timing"][tag])
    json.dump(REP, open(f"{OUT}/info_{'_'.join(JOBS)}.json", "w"), indent=1, default=str)

# finite-value checks
for p in glob.glob(f"{OUT}/**/*.npz", recursive=True):
    z = np.load(p, allow_pickle=True)
    REP["checks"][os.path.basename(p)] = {k: int((~np.isfinite(z[k])).sum()) for k in z.files if z[k].dtype.kind == "f"}
json.dump(REP, open(f"{OUT}/info_{'_'.join(JOBS)}.json", "w"), indent=1, default=str)
log("REPORT", json.dumps({k: v for k, v in REP.items() if k != "checks"}, default=str)[:3000])
log("done")
