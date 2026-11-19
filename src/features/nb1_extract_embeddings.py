# %% [markdown]
# # FLAG 2027 — Notebook 1: extract face + voice embeddings
#
# Uses **the same encoders as your VoxCeleb notebook**, so both embedding sets live in the same space:
# - Face: InsightFace `buffalo_l` (ArcFace, 512-d, `normed_embedding` of the largest detected face)
# - Voice: SpeechBrain `spkrec-ecapa-voxceleb` (ECAPA-TDNN, 192-d) + WavLM-SV (`microsoft/wavlm-base-plus-sv`, 512-d)
# - Face attributes: InsightFace age + gender
#
# **Kaggle settings:** Accelerator = GPU T4, Internet = On.
# **Inputs:** the FLAG (MAV-Celeb v4) dataset with the raw `.jpg`/`.wav` files, plus the training gender meta file.
# Optional secret `HF_TOKEN` (Add-ons → Secrets) for faster Hugging Face downloads.
#
# **Output:** `/kaggle/working/flag_emb/` → `train.npz`, one `<phase>__<track>__<lang>.npz` per test list, and `meta/`.
# After it finishes: **Save Version → Save & Run All**, then add this notebook's output as an input to Notebook 2.
#
# In the evaluation phase (11–18 Nov), add the evaluation data as an extra input and re-run: any test list whose
# path contains "eval" is saved with the `eval__` prefix.

# %%
# Install (same recipe that worked in your VoxCeleb notebook)
import subprocess, sys

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-800:] if r.returncode else "")

sh(f"{sys.executable} -m pip install -q insightface speechbrain opencv-python-headless soundfile")
sh(f"{sys.executable} -m pip uninstall -y onnxruntime onnxruntime-gpu")
sh("rm -rf /usr/local/lib/python3*/dist-packages/onnxruntime*")
sh(f"{sys.executable} -m pip install -q --no-cache-dir onnxruntime-gpu==1.26.0")
sh(f"""{sys.executable} -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())" """)

# %%
import os, glob, json, time, shutil
from collections import Counter
import numpy as np, cv2, torch, torchaudio

INPUT_ROOT = "/kaggle/input"
OUT = "/kaggle/working/flag_emb"
os.makedirs(f"{OUT}/meta", exist_ok=True)

# ---- locate the training set (walks faces/English/idXXX/*.jpg and pairs with voices/.../same name.wav)
hits = glob.glob(f"{INPUT_ROOT}/**/faces/English", recursive=True)
hits = [h for h in hits if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(h)), "voices", "English"))
        and any(d.startswith("id") for d in os.listdir(h))]
assert hits, "Could not find train_set/.../faces/English with id folders. Is the FLAG dataset attached?"
TRAIN_ROOT = os.path.dirname(os.path.dirname(hits[0]))
TRAIN_FACES, TRAIN_VOICES = f"{TRAIN_ROOT}/faces/English", f"{TRAIN_ROOT}/voices/English"
print("train root:", TRAIN_ROOT)

# ---- locate every test pair list: .../gender/English_test.txt, .../no_gender/Bangla_test.txt, ...
test_txts = sorted(t for t in glob.glob(f"{INPUT_ROOT}/**/*_test.txt", recursive=True)
                   if os.path.basename(os.path.dirname(t)) in ("gender", "no_gender"))

def set_name(txt):
    track = os.path.basename(os.path.dirname(txt))           # gender / no_gender
    lang = os.path.basename(txt).split("_")[0]                # English / Bangla
    phase = "eval" if "eval" in txt.lower() else "dev"
    return f"{phase}__{track}__{lang}"

for t in test_txts:
    print(f"{set_name(t):30s} <- {t}")
names = [set_name(t) for t in test_txts]
assert len(names) == len(set(names)), "Two test lists map to the same name; check the paths printed above."

# ---- copy any meta file (training gender annotations) so Notebook 2 finds it
for ext in ("csv", "txt", "tsv", "xlsx", "xls"):
    for p in glob.glob(f"{INPUT_ROOT}/**/*meta*.{ext}", recursive=True) + \
             glob.glob(f"{INPUT_ROOT}/**/*gender*.{ext}", recursive=True):
        if os.path.isfile(p) and not p.endswith("_test.txt"):
            shutil.copy(p, f"{OUT}/meta/{os.path.basename(p)}")
print("meta files copied:", os.listdir(f"{OUT}/meta") or "NONE — attach the gender meta file and re-run this cell")

# %%
# Load the encoders (identical to the VoxCeleb notebook)
import onnxruntime as ort
try:
    ort.preload_dlls()
except Exception as e:
    print("preload:", e)
print("onnxruntime providers:", ort.get_available_providers())

from insightface.app import FaceAnalysis
from speechbrain.inference.speaker import EncoderClassifier
from huggingface_hub import snapshot_download

token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = token
except Exception:
    print("No HF_TOKEN secret; downloading anonymously.")

# detection (alignment keypoints) + recognition (ArcFace) + genderage (age / gender attributes).
# cudnn_conv_algo_search=DEFAULT avoids the cuDNN-frontend "No valid engine configs" failure on T4 (keeps models on GPU).
FACE_PROVIDERS = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition", "genderage"],
                        providers=FACE_PROVIDERS)
face_app.prepare(ctx_id=0, det_size=(640, 640))
face_rec = face_app.models["recognition"]
print("face model providers:", {k: m.session.get_providers()[0] for k, m in face_app.models.items()})

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
# Use the exact ECAPA copy saved with your VoxCeleb embeddings if it is attached; otherwise download it.
local = glob.glob(f"{INPUT_ROOT}/**/ecapa_model/hyperparams.yaml", recursive=True)
if local:
    ecapa_dir = "/kaggle/working/ecapa_model"
    shutil.copytree(os.path.dirname(local[0]), ecapa_dir, dirs_exist_ok=True)   # input is read-only
    print("ECAPA from attached dataset:", os.path.dirname(local[0]))
else:
    ecapa_dir = snapshot_download("speechbrain/spkrec-ecapa-voxceleb", token=token,
                                  local_dir="/kaggle/working/ecapa_model")
voice_model = EncoderClassifier.from_hparams(source=ecapa_dir, savedir=ecapa_dir, run_opts={"device": dev})
from transformers import AutoFeatureExtractor, WavLMForXVector
wl_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
wl_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()

# %%
def face_embedding(path):
    """Largest detected face -> (normed_embedding, status, age, male). Pads tight crops; falls back to a raw crop."""
    img = cv2.imread(path)
    if img is None:
        return None, "unreadable", np.nan, -1
    for pad in (0.0, 0.5):
        im = img
        if pad:
            ph, pw = int(img.shape[0] * pad), int(img.shape[1] * pad)
            im = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        faces = face_app.get(im)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return (f.normed_embedding.astype(np.float32), ("det" if not pad else "det_pad"),
                    float(f.age), int(f.sex == "M"))
    e = face_rec.get_feat([cv2.resize(img, (112, 112))]).ravel()
    return (e / np.linalg.norm(e)).astype(np.float32), "fallback", np.nan, -1


def load_wav(path):
    try:
        sig, sr = torchaudio.load(path)
    except Exception:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        sig = torch.from_numpy(x.T.copy())
    return sig, sr


def voice_embedding(path):
    """(ECAPA raw 192-d, WavLM-SV 512-d) on 16 kHz mono, full clip — same encoders as the VoxCeleb extraction."""
    sig, sr = load_wav(path)
    sig = sig.mean(0, keepdim=True)
    if sr != 16000:
        sig = torchaudio.functional.resample(sig, sr, 16000)
    if sig.shape[1] < 16000:                       # very short clip: repeat to at least 1 s
        sig = sig.repeat(1, int(np.ceil(16000 / max(sig.shape[1], 1))))
    with torch.no_grad():
        e = voice_model.encode_batch(sig.to(dev)).squeeze().cpu().numpy().astype(np.float32)
        inp = wl_fe(sig[0].numpy(), sampling_rate=16000, return_tensors="pt")
        w = wl_model(**{k: v.to(dev) for k, v in inp.items()}).embeddings.squeeze().cpu().numpy().astype(np.float32)
    return e, w

# %%
# Training set
t0 = time.time()
rows = []
for spk in sorted(d for d in os.listdir(TRAIN_FACES) if os.path.isdir(f"{TRAIN_FACES}/{d}")):
    for jpg in sorted(glob.glob(f"{TRAIN_FACES}/{spk}/*.jpg")):
        stem = os.path.basename(jpg)[:-4]
        wav = f"{TRAIN_VOICES}/{spk}/{stem}.wav"
        if os.path.exists(wav):
            rows.append((spk, stem, jpg, wav))
print("training pairs found:", len(rows), "| identities:", len({r[0] for r in rows}))

F_, V_, W_, A_, M_, S_, ST_, STAT = [], [], [], [], [], [], [], []
for i, (spk, stem, jpg, wav) in enumerate(rows):
    f, status, age, male = face_embedding(jpg)
    if f is None:
        continue
    e, w = voice_embedding(wav)
    F_.append(f); V_.append(e); W_.append(w); A_.append(age); M_.append(male)
    S_.append(spk.lower()); ST_.append(stem); STAT.append(status)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(rows)}  ({(time.time()-t0)/60:.1f} min)")

np.savez(f"{OUT}/train.npz", face=np.stack(F_), voice=np.stack(V_), wavlm=np.stack(W_), age=np.array(A_, np.float32),
         male=np.array(M_, np.int8), spk=np.array(S_), stem=np.array(ST_), face_status=np.array(STAT))
print("train.npz:", np.stack(F_).shape, np.stack(V_).shape, Counter(STAT))

# %%
# Test pair lists (dev now; eval too when it is attached)
report = {"train": dict(Counter(STAT))}
for txt in test_txts:
    name = set_name(txt)
    root = os.path.dirname(txt)
    lines = [l.split() for l in open(txt) if l.strip()]
    pid = [l[0] for l in lines]
    fcache, vcache, stat = {}, {}, Counter()
    Fx, Vx, Wx, Ax, Mx = [], [], [], [], []
    for _, vrel, frel in (l[:3] for l in lines):
        fp, vp = os.path.join(root, frel), os.path.join(root, vrel)
        if fp not in fcache:
            e, s, age, male = face_embedding(fp)
            if e is None:                            # unreadable image: zero vector -> neutral score later
                e, s = np.zeros(512, np.float32), "unreadable"
            fcache[fp] = (e, age, male); stat[s] += 1
        if vp not in vcache:
            vcache[vp] = voice_embedding(vp)
        Fx.append(fcache[fp][0]); Ax.append(fcache[fp][1]); Mx.append(fcache[fp][2])
        Vx.append(vcache[vp][0]); Wx.append(vcache[vp][1])
    np.savez(f"{OUT}/{name}.npz", pid=np.array(pid), face=np.stack(Fx), voice=np.stack(Vx), wavlm=np.stack(Wx),
             age=np.array(Ax, np.float32), male=np.array(Mx, np.int8))
    report[name] = dict(stat)
    print(f"{name}: {len(pid)} pairs  faces {dict(stat)}  ({(time.time()-t0)/60:.1f} min)")

json.dump(report, open(f"{OUT}/face_detection_report.json", "w"), indent=1)

# %%
print("Output files:")
for p in sorted(glob.glob(f"{OUT}/**", recursive=True)):
    if os.path.isfile(p):
        print(f"  {os.path.relpath(p, OUT):45s} {os.path.getsize(p)/1e6:7.2f} MB")
shutil.rmtree("/kaggle/working/ecapa_model", ignore_errors=True)   # keep the output small
