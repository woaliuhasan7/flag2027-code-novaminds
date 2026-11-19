"""Build the AdaFace-in-the-VGGFace2-slot input folder (dataset mahiwoaliuhasan/flag-ada-vgg) for models M3 / M4.
Every VGGFace2 file is copied key by key and only 'face_vgg' is replaced by the AdaFace IR-101 embedding of the SAME face crop
(same row: FLAG by spk/stem, VoxCeleb2 by row order = (clip, frame #) join done in the extraction). The ViT age / gender fields,
detection status, clip and speaker ids are unchanged, so the age input and the Stage-A rows are identical to M1 / M2.
Rows whose AdaFace crop is missing (NaN) would be marked status 'nodet' (dropped like VGGFace2 non-detections) and counted."""
import glob, json, os, sys
import numpy as np
SRC_VOX = sys.argv[1]                      # folder with adaface_vox/<tag>.npz from the five shahiraalima kernels
OUT = "/home/claude/advanced_experiments/kaggle/ds_ada"; V11 = "/home/claude/kaggle/v11_kin"
ADA_FLAG = "/home/claude/kaggle/xj1/adaface_dl/adaface/flag/train.npz"
os.makedirs(f"{OUT}/flag_vgg", exist_ok=True); REP = {"flag": {}, "vox": {}}
v = np.load(f"{V11}/vgg1/flag_vgg/train.npz", allow_pickle=True); a = np.load(ADA_FLAG, allow_pickle=True)
assert (v["spk"].astype(str) == a["spk"].astype(str)).all() and (v["stem"].astype(str) == a["stem"].astype(str)).all()
assert np.isfinite(a["adaface"]).all()
np.savez(f"{OUT}/flag_vgg/train.npz", **{k: (a["adaface"].astype(np.float32) if k == "face_vgg" else v[k]) for k in v.files})
REP["flag"] = {"n": int(len(a["adaface"])), "status": {s: int((a["status"] == s).sum()) for s in np.unique(a["status"])}}
for vf in sorted(glob.glob(f"{V11}/vgg*/vox_vgg/*/vgg.npz")):
    tag = os.path.basename(os.path.dirname(vf)); v = np.load(vf, allow_pickle=True)
    hits = glob.glob(f"{SRC_VOX}/**/adaface_vox/{tag}.npz", recursive=True); assert len(hits) == 1, (tag, hits)
    z = np.load(hits[0], allow_pickle=True); n = len(v["clip"])
    assert len(z["adaface"]) == n and (z["clip"].astype(str) == v["clip"].astype(str)).all() and (z["spk"].astype(str) == v["spk"].astype(str)).all(), tag
    E = z["adaface"].astype(np.float32); bad = ~np.isfinite(E).all(1); st = v["status"].copy()
    newly = int((bad & (st != "nodet")).sum()); st[bad] = "nodet"; E[bad] = 0.0
    os.makedirs(f"{OUT}/vox_vgg/{tag}", exist_ok=True)
    np.savez(f"{OUT}/vox_vgg/{tag}/vgg.npz", **{k: (E if k == "face_vgg" else st if k == "status" else v[k]) for k in v.files})
    REP["vox"][tag] = {"rows": n, "missing_adaface_rows": int(bad.sum()), "rows_dropped_beyond_vgg_nodet": newly,
                       "adaface_status": {s: int((z["status"] == s).sum()) for s in np.unique(z["status"])}}
    print(tag, REP["vox"][tag], flush=True)
json.dump(REP, open(f"{OUT}/assembly_report.json", "w"), indent=1)
json.dump({"title": "flag-ada-vgg", "id": "mahiwoaliuhasan/flag-ada-vgg", "licenses": [{"name": "other"}]}, open(f"{OUT}/dataset-metadata.json", "w"))
print("done", json.dumps(REP)[:800])
