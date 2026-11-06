# Legacy cells in `vit_train_from_scratch_{5gest,5part,10part}`

These three notebooks keep ~28 scratch cells after the markdown cell "Legacy scratch cells below". They were
never updated with the seeding / train-only normalisation / validation-split fixes and are kept only for the
record (nothing was deleted). Cell indices are 0-based and refer to `vit_train_from_scratch_5gest`; the other
two notebooks hold the same blocks shifted by a small offset (last section).

"Cover" means `src/emg_gesture` (CLI `emg-gesture svm|vit-google|vit-scratch`) reproduces the idea in its fixed
form, not that the old cell's numbers are reproducible.

| Cells (5gest) | What the block did | Covered? |
|---|---|---|
| 6 | Augmentations `add_noise`, `time_warp`, `scale_signal`, `random_crop_pad`, `flip`, and `augment_data` combining them. | cover: `emg_gesture/augment.py` (Generator-based, seeded) for all but `flip`; `flip` not covered (it was not used by the pipeline in the package) |
| 7 | Full scratch-ViT run: `.mat` loading with NaN removal, pad/truncate to 10240 samples, reshape to (3,224,224), label remap to 0..K-1, stratified 80/20 split, `ViTConfig` (hidden 128, 4 layers, 4 heads, patch 16, dropout 0.3), `DataParallel`, AdamW 1e-4, warm-up + cosine scheduler stepped per batch, grad-clip 1.0, per-epoch "validation" on the test set. | cover: `data.py`, `models.py`, `train_vit.py`, `vit-scratch` (with validation split, train-only normalisation, test evaluated once). Multi-GPU `DataParallel` not covered. |
| 8 | `torch.save(state_dict, "best_model_5gest")`. | cover (best-validation checkpoint in the run dir) |
| 9, 10, 14 | Re-definitions of `EMGDataset`, `load_and_process_file` and the split without augmentation. | cover (superseded by `data.py`) |
| 11, 12, 13 | Earlier scratch ViT: `ViTConfig` hidden 512, 6 layers, 4 heads, intermediate 1024, patch 16, dropout 0.3, AdamW 3e-4 wd 1e-4, 30 epochs (cell 13: a 50-epoch loop), test accuracy printed each epoch. | not covered as that hyper-parameter set (the pipeline is covered; the config is not shipped, override with `--set`) |
| 15, 19, 22 | `ViTFeatureExtractor` + `google/vit-base-patch16-224-in21k`, `num_labels` 5 / 5 / 10, AdamW. | cover: `vit-google` |
| 16, 20, 23 | 20-epoch training loops for those pre-trained ViTs. | cover: `vit-google` |
| 17, 18, 21 | Label parser plus dataset setup for the google-ViT runs. | cover: `data.py` |
| 24, 25 | SVM exploration: flatten padded `.mat`, split, on the `s1_s2_s3_matfiles_20part` folder. | cover for the SVM idea (`svm`); the `20part` task has no config: not covered |
| 26 | `PCA(n_components=2000)` to find the number of components for 95 % explained variance. | not covered (belongs to `pca_components_search.ipynb`, left in place) |
| 27 | PCA + default `SVC()`, test accuracy. | cover: `svm` (Pipeline of scaler, PCA, SVC; PCA fitted inside the folds) |
| 28 | PCA(100)+SVC on the three raw `SessionN_matfiles` folders merged. | not covered (merging raw session folders is a data-prep step; see the quirks in `scripts/prepare_folders.py`) |
| 29 | `SGDClassifier` with batched `partial_fit` (PCA re-fitted on every batch, a bug) and a `GridSearchCV` run on the test set. | not covered (no SGD variant, no grid search) |
| 30, 32, 34 | Helper fragments: label parser, padding helper, commented NaN/Inf check. | cover (`data.py`) |
| 31, 33 | PCA(50) + linear `SVC`, with a comment about feeding PCA output to a ViT. | partly: linear kernel via `--set svm.kernel=linear`; PCA features into a ViT was never implemented: not covered |

## Offsets in the other notebooks

* `vit_train_from_scratch_5part`: same blocks; from cell 8 on, indices are one lower (PCA+SVC = 27, SGD/grid = 28).
  Cell 7 additionally saves `vit_emg_model.pth`.
* `vit_train_from_scratch_10part`: no augmentation cell, so the scratch run is cell 6 and later cells are two lower
  than in 5gest (SGD/grid = 27). In the `*part*` notebooks the label is the participant id, not the gesture.

## Conclusion

The main path (loading, augmentation, scratch ViT, google ViT, PCA+SVC, checkpointing) is covered by the package
in a fixed form. Not covered: `DataParallel`, `flip`, the hidden-512/6-layer scratch config and its 30/50-epoch
loops, the `20part` folder, the PCA-95%-variance search (in `pca_components_search`), the SGDClassifier +
GridSearchCV experiment, merging raw session folders, and PCA features fed to a ViT (never implemented).
