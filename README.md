# FLAG 2027 — team NovaMinds code (best evaluation submission)

Submission reproduced here: `outputs/submission/submission_final_20p33_fliptta_adaface.zip`
(SHA-256 `3578b07558246eeaca7303d427c6140609a3ca05917d45b31dfe1a4aae62ed41`).

## System
Score = (4 · v_flip − s_ada) / 5 per trial (lower = same person; plain cosine, each trial scored on its own).
Five equal-weight members:
| # | Member | Face input | Voice input |
|---|---|---|---|
| 1 | `rdn_aux` | VGGFace2 InceptionResnetV1 (+ ViT age), horizontal-flip TTA | ReDimNet2-B3 |
| 2 | `vb2_aux` | VGGFace2 (+ age), flip TTA | VoxBlink2 + WavLM-SV |
| 3 | `clip_aux` | VGGFace2 + CLIP ViT-B/16 (+ age), flip TTA on the VGGFace2 part | ECAPA + WavLM-SV |
| 4 | `xj0` (mean of 3 runs) | fine-tuned InceptionResnetV1 | fine-tuned WavLM-SV |
| 5 | AdaFace + CAM++ | AdaFace IR-101 (WebFace12M) in the face slot (+ age) | CAM++ (official) |
Members 1-4 form the earlier 20.33 ensemble (0.25 each; xj0 runs 1/12 each); the flip TTA replaces the stored VGGFace2 embedding
by l2n(stored + l2n(VGGFace2(mirrored crop))) for members 1-3. Towers: MLP 256-256-128 per modality, trained on VoxCeleb2
(Bengali speakers removed, Stage A) then the 70 English FLAG training identities (Stage B), 3 seeds. Predicted gender is not an input
(age only); gender only composes training batches. No score normalisation, no cohort/test statistics, no clustering/graphs.

## Data (Kaggle, public)
Frozen-encoder features and inputs used by the scripts (`https://www.kaggle.com/datasets/mahiwoaliuhasan/<name>`):
| Dataset | Content |
|---|---|
| [flag-v11-in](https://www.kaggle.com/datasets/mahiwoaliuhasan/flag-v11-in) | main features (VGGFace2, CLIP, ECAPA, WavLM, VoxBlink2, ViT age), language-ID, xj0 embeddings, `nb2v11.py` |
| [flag-ho-extra](https://www.kaggle.com/datasets/mahiwoaliuhasan/flag-ho-extra) | extra inputs (ReDimNet2 features, xj0 sg50 embeddings, Bengali speaker list) |
| [flag-rdn-b3](https://www.kaggle.com/datasets/mahiwoaliuhasan/flag-rdn-b3) | ReDimNet2-B3 features (FLAG + VoxCeleb2) |
| [flag-ada-vgg](https://www.kaggle.com/datasets/mahiwoaliuhasan/flag-ada-vgg) | AdaFace IR-101 features (FLAG train + VoxCeleb2) |
| [flag-ada-eval](https://www.kaggle.com/datasets/mahiwoaliuhasan/flag-ada-eval) | AdaFace features of the 4 evaluation lists |
| [flag2027-campp-feats](https://www.kaggle.com/datasets/mahiwoaliuhasan/flag2027-campp-feats) | CAM++ features (FLAG train, VoxCeleb2, evaluation lists) |
| [flag-tta-in](https://www.kaggle.com/datasets/mahiwoaliuhasan/flag-tta-in) | flip-TTA features, production towers, `tta_prod.py` |
| [flag-adv-code](https://www.kaggle.com/datasets/mahiwoaliuhasan/flag-adv-code) | code used by the AdaFace/CAM++ kernel |

## Layout
- `src/towers/nb2v11.py` tower training/scoring (members 1-3, 5); `configs/final_config_ageonly_rdn.json` members/weights.
- `src/xj0/xft_full.py` xj0 encoder fine-tuning (PC script, N-way loss, query pooling); `PC_README.txt`, `*.ps1` launch scripts.
- `src/adaface_campp/train_adaface_campp.py` member 5 training + evaluation cosines.
- `src/features/` frozen-encoder feature extraction (`nb1_extract_embeddings.py`, flip features, AdaFace/CAM++ extraction).
- `src/final/tta_prod.py` production 20.33 + flip TTA (checks no-flip scores reproduce the 20.33 files before writing);
  `src/final/combine_final.py` builds the final zip.
- `checkpoints/towers/` 18 trained towers (3 seeds × rdn/vb2/clip + 3 xj0 runs × 3 seeds); `checkpoints/adaface_campp/` 3 seeds.
- `docs/architecture.png` system architecture diagram (source `docs/architecture.mmd`).
- `outputs/` per-member eval scores, flip features, eval cosines, and the submitted zips.

## Reproduce
1. Final combination (verified: byte-identical score files to the submitted zip):
   `python src/final/combine_final.py outputs/submission/submission_final_20p33_fliptta.zip outputs/eval_cosines_M4ADA.npz out.zip`
2. Towers from features: `nb2v11.py` (COMPLIANT=1, 12 epochs Stage A, 15 epochs Stage B, seeds 0/1/2, see `tta_prod.py` / `train_adaface_campp.py` for the exact launch).
3. Features come from the extraction scripts in `src/features/` (Kaggle notebooks); see KNOWN_GAPS.md.
See `requirements.txt`.
