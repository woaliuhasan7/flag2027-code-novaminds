"""Adds the AdaFace IR-101 + CAM++ member (production tower recipe, all 70 ids, seeds 0/1/2, plain cosine averaged over seeds) with
equal member weights fixed before evaluation, to two existing submissions:
 A) 7 members, 1/7 each: production 4 (rdn, vb2, clip, xj0 x3 averaged) + CLIP-xj0 + VGGFace2/CAM++ + AdaFace/CAM++
    file = (4 v_prod - s_clipxj0 - s_campp - s_ada) / 7
 B) 5 members, 1/5 each: 20.33 + flip-TTA (rdn, vb2, clip with flip, xj0 x3 averaged) + AdaFace/CAM++
    file = (4 v_flip - s_ada) / 5
v = submitted value of the base file (= -0.25 * sum of its four member scores). Lower = same person. Each trial scored on its own."""
import sys, zipfile, json, hashlib, numpy as np
PROD, FLIP, CAM, RAW, ADA, OUTA, OUTB = sys.argv[1:8]
N = {"no_gender/sub_score_v4_English_heard.txt": ("no_gender", "English", 392), "no_gender/sub_score_v4_Bangla_unheard.txt": ("no_gender", "Bangla", 802),
     "gender/sub_score_v4_English_heard.txt": ("gender", "English", 416), "gender/sub_score_v4_Bangla_unheard.txt": ("gender", "Bangla", 716)}
def rd(z, fn):
    r = [l.split() for l in zipfile.ZipFile(z).read(fn).decode().strip().split("\n")]; return [x[0] for x in r], np.array([float(x[1]) for x in r])
cam = np.load(CAM); ada = np.load(ADA, allow_pickle=True); FA, FB, rep = {}, {}, {}
for fn, (tr, lg, n) in N.items():
    pid, v = rd(PROD, fn); pf, vf = rd(FLIP, fn); assert pf == pid and len(pid) == len(set(pid)) == n
    r = np.load(f"{RAW}/raw_{tr}__{lg}.npz", allow_pickle=True); assert [str(x) for x in r["pid"]] == pid; sc = r["s"].astype(np.float64)
    sm = cam[f"eval__{tr}__{lg}"].astype(np.float64)
    k = f"eval__{tr}__{lg}"; assert [str(x) for x in ada[f"{k}__pid"]] == pid
    sa = np.mean([ada[f"{k}__seed{s}"] for s in (0, 1, 2)], 0).astype(np.float64)
    a = (4 * v - sc - sm - sa) / 7; b = (4 * vf - sa) / 5
    for x in (v, vf, sc, sm, sa, a, b): assert len(x) == n and np.isfinite(x).all()
    FA[fn] = "".join(f"{p} {x:.6f}\n" for p, x in zip(pid, a)); FB[fn] = "".join(f"{p} {x:.6f}\n" for p, x in zip(pid, b))
    rk = lambda q: np.argsort(np.argsort(q))
    rep[fn] = dict(rows=n, corr_ada_vs_prod=float(np.corrcoef(-v, sa)[0, 1]), spearman_A_vs_prod=float(np.corrcoef(rk(v), rk(a))[0, 1]),
                   spearman_B_vs_prod=float(np.corrcoef(rk(v), rk(b))[0, 1]))
for OUT, F in ((OUTA, FA), (OUTB, FB)):
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for d in ("gender/", "no_gender/"): z.writestr(d, "")
        for fn, t in F.items(): z.writestr(fn, t)
    z = zipfile.ZipFile(OUT); assert sorted(x for x in z.namelist() if x.endswith(".txt")) == sorted(N)
    for fn, (tr, lg, n) in N.items():
        p2, v2 = rd(OUT, fn); assert len(p2) == n and p2 == rd(PROD, fn)[0] and np.isfinite(v2).all()
    rep[OUT.split("/")[-1]] = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
print(json.dumps(rep, indent=1)); print("VALIDATED", OUTA, OUTB)
