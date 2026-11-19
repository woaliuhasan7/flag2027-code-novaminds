# FLAG 2027 TTA - VGGFace2 (facenet-pytorch InceptionResnetV1 'vggface2') embeddings of the cached 160x160 face crops, original and
# horizontally flipped, for FLAG train + the 4 official evaluation lists. Same preprocessing as the production extraction:
# BGR->RGB, resize 160, (x - 127.5) / 128, L2-normalised. Each image is processed on its own (no cross-sample statistics).
import glob, json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "facenet-pytorch==2.6.0", "--no-deps"], check=False)
import numpy as np, cv2, torch
from facenet_pytorch import InceptionResnetV1
torch.set_num_threads(4); T0 = time.time()
vgg = InceptionResnetV1(pretrained="vggface2").eval()
OUT = "/kaggle/working/face"; os.makedirs(OUT, exist_ok=True); REP = {}
@torch.no_grad()
def emb(imgs):
    out = []
    for a in range(0, len(imgs), 64):
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in imgs[a:a + 64]]
        x = torch.from_numpy(np.stack([cv2.resize(c, (160, 160)) for c in rgb])).permute(0, 3, 1, 2).float()
        out.append(torch.nn.functional.normalize(vgg((x - 127.5) / 128.0), dim=-1).numpy())
    return np.concatenate(out)
for p in sorted(glob.glob("/kaggle/input/**/flag-tta-crops/**/*.npz", recursive=True)):
    name = os.path.basename(p); c = np.load(p, allow_pickle=True); t = time.time()
    imgs = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in c["jpg"]]
    e0 = emb(imgs); e1 = emb([np.ascontiguousarray(im[:, ::-1]) for im in imgs])
    keep = {k: c[k] for k in c.files if k != "jpg"}
    np.savez(f"{OUT}/{name}", vgg_orig=e0.astype(np.float32), vgg_flip=e1.astype(np.float32), **keep)
    REP[name] = {"n": len(imgs), "seconds": time.time() - t, "mean_cos_orig_flip": float((e0 * e1).sum(1).mean())}; print(name, REP[name], flush=True)
json.dump({"torch": torch.__version__, "timing": REP, "minutes": (time.time() - T0) / 60}, open(f"{OUT}/info.json", "w"), indent=1)
print("done", flush=True)
