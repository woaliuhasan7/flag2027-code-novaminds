# Known gaps (to be resolved before final hand-over)
- The fine-tuned xj0 encoder weights (InceptionResnetV1 + WavLM) were trained on a lab PC and are not yet in this repo; only their trainer
  (`src/xj0/xft_full.py`) and the resulting towers/embeddings used downstream are. Weights will be provided by link.
- Large feature datasets (VoxCeleb/FLAG frozen-encoder features, ~1 GB+) are on private Kaggle datasets, not in git.
- `xft_full.py` is the current PC script; the production xj0 runs were made with the Nov-13 version, which differs in two lines
  (feature-folder glob and a `no_vox_feats` option) that do not change training.
- Pretraining-data statements for each public encoder are listed in the system description, not here.
