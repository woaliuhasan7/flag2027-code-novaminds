# Known gaps (to be resolved before final hand-over)
- The fine-tuned xj0 encoder weights (InceptionResnetV1 + WavLM, 3 final runs, ~526 MB each) are too large for git. They are in the public Kaggle dataset
  `mahiwoaliuhasan/flag2027-xj0-encoders` (SHA-256 in that dataset's `SHA256SUMS`). Trainer: `src/xj0/xft_full.py`.
- Large feature datasets (VoxCeleb/FLAG frozen-encoder features, ~1 GB+) are on private Kaggle datasets, not in git.
- `xft_full.py` is the current PC script; the production xj0 runs were made with the Nov-13 version, which differs in two lines
  (feature-folder glob and a `no_vox_feats` option) that do not change training.
- Pretraining-data statements for each public encoder are listed in the system description, not here.
