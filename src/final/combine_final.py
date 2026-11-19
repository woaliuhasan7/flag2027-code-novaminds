"""Final FLAG 2027 submission = 5 equal-weight members, plain cosine, lower = same person.
  4 x v_flip : production 20.33 ensemble (rdn_aux, vb2_aux, clip_aux with horizontal-flip face TTA, xj0 = mean of 3 runs), file value
               v_flip = -0.25 * sum of its four member scores (written by tta_prod.py)
  1 x s_ada  : AdaFace IR-101 + CAM++ tower member (train: src/adaface_campp/train_adaface_campp.py), mean over seeds 0/1/2
  file = (4*v_flip - s_ada) / 5          (each trial scored on its own; no normalisation, no cross-trial statistics)
Usage: python combine_final.py outputs/submission/submission_final_20p33_fliptta.zip outputs/eval_cosines_M4ADA.npz OUT.zip"""
import sys, zipfile, numpy as np
FLIP, ADA, OUT = sys.argv[1:4]
N = {"no_gender/sub_score_v4_English_heard.txt": ("no_gender", "English", 392), "no_gender/sub_score_v4_Bangla_unheard.txt": ("no_gender", "Bangla", 802),
     "gender/sub_score_v4_English_heard.txt": ("gender", "English", 416), "gender/sub_score_v4_Bangla_unheard.txt": ("gender", "Bangla", 716)}
def rd(z, fn):
    r = [l.split() for l in zipfile.ZipFile(z).read(fn).decode().strip().split("\n")]; return [x[0] for x in r], np.array([float(x[1]) for x in r])
ada = np.load(ADA, allow_pickle=True); F = {}
for fn, (tr, lg, n) in N.items():
    pid, vf = rd(FLIP, fn); k = f"eval__{tr}__{lg}"
    assert len(pid) == len(set(pid)) == n and [str(x) for x in ada[f"{k}__pid"]] == pid
    sa = np.mean([ada[f"{k}__seed{s}"] for s in (0, 1, 2)], 0).astype(np.float64)
    b = (4 * vf - sa) / 5; assert np.isfinite(b).all()
    F[fn] = "".join(f"{p} {x:.6f}\n" for p, x in zip(pid, b))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ("gender/", "no_gender/"): z.writestr(d, "")
    for fn, t in F.items(): z.writestr(fn, t)
print("wrote", OUT)
