# archive: superseded notebooks

These are the original thesis experiment notebooks. They are superseded by the `emg_gesture` package
(`src/emg_gesture`, configs in `configs/`) and are kept for provenance only. Outputs were stripped and they
have not been re-run since the review fixes (see the top-level README, "Known limitations"). Do not extend
them; use the CLI.

| Notebook(s) | Replacement |
|---|---|
| `folders_creation.ipynb` | `python scripts/prepare_folders.py --source-dir SRC --dest-dir DEST --task all` |
| `svm_10gest.ipynb` | `emg-gesture svm --task 10gest` |
| `svm_5gest.ipynb` | `emg-gesture svm --task 5gest` |
| `svm_17participants.ipynb` | `emg-gesture svm --task 17participants` |
| `svm_10part.ipynb` | `emg-gesture svm --task 10part` |
| `svm_5part.ipynb` | `emg-gesture svm --task 5part` |
| `vit_google_<task>.ipynb` | `emg-gesture vit-google --task <task>` |
| `vit_train_from_scratch_<task>.ipynb` | `emg-gesture vit-scratch --task <task>` |

`<task>` is one of `10gest`, `5gest`, `17participants`, `10part`, `5part`. The legacy cells at the end of
`vit_train_from_scratch_{5gest,5part,10part}` are documented in [LEGACY_NOTES.md](LEGACY_NOTES.md).
`code_reproduction_max_samples`, `pca_components_search` and `mlflow_plotting` remain in `03_experiments/`.
