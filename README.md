# vit-thesis

Code for my MSc thesis, *Vision Transformer for Gesture Recognition from EMG Signals* (MSc Digital Media & Computational Intelligence, Aristotle University of Thessaloniki): Vision Transformers (fine-tuned and trained from scratch) compared against SVM baselines across gesture-count and participant splits. The repository also holds a separate pipeline for ViT-GPT2 image captioning on a spatial COCO dataset.

> TODO: one sentence on the research question and the headline result.

## Repository layout

```
01_dataset/      captioning side work (not the EMG thesis): tester.py (RGB+depth experiment),
                 notebooks (vitgpt2.ipynb, test_gpt.ipynb); archive/ = scripts superseded by src/captioning
02_training/     captioning side work: training_version.ipynb; archive/train.py superseded by src/captioning
src/captioning/  captioning side package: `captioning subset|remove-empty|train|infer`
                 (configs/captioning.yaml; DATA_DIR / OUTPUT_DIR env or --data-dir / --output-dir)
03_experiments/  the EMG thesis experiments
  code_reproduction_max_samples.ipynb
  pca_components_search.ipynb      PCA analysis
  mlflow_plotting.ipynb            result plots from MLflow
  archive/                         original folders_creation / svm_* / vit_google_* /
                                   vit_train_from_scratch_* notebooks, superseded by the package
                                   (see archive/README.md and archive/LEGACY_NOTES.md)
scripts/prepare_folders.py         builds the per-task dataset folders (port of folders_creation)
src/emg_gesture/, configs/, tests/ package, configs and tests
requirements.txt         direct dependencies
requirements-freeze.txt  full original environment freeze
```

The `captioning` commands take `--data-dir` (or `DATA_DIR`) for the dataset root and `--output-dir` (or `OUTPUT_DIR`). Outputs were stripped from the large notebooks to keep the repository small, so re-run them to regenerate results.

Notebook suffixes mean the experimental split: `5gest` / `10gest` (5 or 10 gesture classes), `5part` / `10part` / `17participants` (number of participants).

## Package and CLI (`src/emg_gesture`)

Two layouts coexist: the original notebooks in `03_experiments/archive/` (self-contained, unchanged) and the
reusable package in `src/emg_gesture/` with configs in `configs/` (`base.yaml` < `svm.yaml` < `task/<task>.yaml`).
Both read the same data layout: one flat folder of `*.mat` per task (e.g. `s1_s2_s3_matfiles_10gest/`), labels parsed
from filenames (raw ids, `remap_labels` gives 0..K-1 for ViT).

```bash
pip install -e .[dev]
emg-gesture svm --task 10gest --split grouped --seed 42 --set data_dir=/path/to/folder --set svm.kernel=rbf
```

Writes `runs/<task>_svm_<seed>_<timestamp>/` (`config.resolved.yaml`, `metrics.json`, `confusion.png`, `per_class.csv`).
`*part` / `17participants` tasks use the participant as label, so the split is forced to `random` (logged).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins direct dependencies without CUDA tags; the original freeze used the CUDA 11.8 builds (`+cu118`) of torch/torchvision. For GPU, install matching wheels from https://download.pytorch.org/whl/cu118 first. `scikit-learn`, `matplotlib` and `mlflow` are imported by the notebooks but were absent from the freeze, so they are unpinned.

## Reproduce

1. Build the dataset (EMG): see `scripts/`. (Captioning side work is independent: `captioning --help`.)
2. Create the task folders: `python scripts/prepare_folders.py --source-dir SRC --dest-dir DATA --task all` (try `--dry-run` first; the script docstring lists the original notebook's open quirks).
3. Run the experiments per task `T` (`10gest 5gest 17participants 10part 5part`): `emg-gesture svm --task T`, then `emg-gesture vit-google --task T`, then `emg-gesture vit-scratch --task T`, each with `--set data_dir=DATA/<task folder>`.
4. Optional: `03_experiments/mlflow_plotting.ipynb` for plots from tracked runs.

## Results

| Model | Split | Accuracy |
|---|---|---|
| TODO | TODO | TODO |

## Data

TODO: where the data lives (not committed), how to obtain it, consent/licensing note.

## Citation

```
TODO: thesis reference
```

## License

MIT, see [LICENSE](LICENSE).

## Known limitations

These points come from a code review of the notebooks; they are not experimental results.

**Important: the fixes below were made in the notebook code only. None of the notebooks has been re-run since, and the stored outputs of the edited cells were cleared. Any accuracy numbers published or reported from earlier runs (thesis text, slides, earlier READMEs) used the old, flawed code and are NOT comparable with what the current notebooks will produce. Re-run everything before quoting a number.**

Fixed in code (not re-run):

- Seeds: every `svm_*` / `vit_*` notebook now starts with a seed cell (`SEED = 42`; `random`, NumPy, torch/CUDA, DataLoader `generator` / `worker_init_fn`). Results still come from a single run, and GPU runs are not guaranteed bit-exact.
- SVM: StandardScaler, PCA and SVC are in one scikit-learn `Pipeline`, and that pipeline is what gets cross-validated, fitted and scored, so PCA is fitted inside each fold (previously `svm_10part` / `svm_5part` / `svm_17participants` computed PCA features but fitted on the raw features, and PCA was fitted before cross-validation). `svm_5gest` and `svm_17participants` previously had no cross-validation; it was added.
- ViT: the per-epoch evaluation computes its own loss with the criterion on each evaluation batch (the `vit_google` notebooks reused the last training loss), train/validation loss are averaged over batches, and train accuracy is computed over the whole epoch.
- ViT: a separate validation split is held out from the training data; checkpoints are selected on validation accuracy; the test set is evaluated once, after training (previously the test set doubled as the validation set every epoch, including for checkpoint selection in `vit_train_from_scratch`).
- `*gest*` notebooks (including the `17participants` ones, whose label is the gesture): a `SPLIT_MODE` variable selects a participant-grouped split (`'grouped'`, default, `GroupShuffleSplit`) or the old random per-file split (`'random'`). The participant ID is parsed from the filename token containing `participant`; this was verified from the parsing code in `folders_creation.ipynb` only, not against the real files, and it assumes the same participant number means the same person across the mixed sessions.
- Normalisation: the per-sample min-max scaling was replaced by per-channel mean/std computed on the training data only and applied to train / validation / test (SVM: per-EMG-channel, inside the pipeline; ViT: per channel of the 3x224x224 array built from the flattened window, with a zero-std guard).
- Final evaluation cells report the confusion matrix, per-class accuracy and chance level (uniform and majority-class) in addition to accuracy.
- The `*part*` notebooks now state at the top that the label is the participant ID (identification task).

Still open:

- The `*part*` notebooks (and the 5part/10part scratch ones) perform participant identification, not gesture recognition, and cannot be split by participant; they use a random per-file split.
- In `*gest*` mode `'random'` the numbers are within-subject and do not measure generalisation to new users.
- Results are single runs; no confidence intervals or repeated seeds.
- The legacy scratch cells that follow the main pipeline in `vit_train_from_scratch_5gest`, `_5part` and `_10part` were not updated (marked in a markdown cell); `folders_creation`, `code_reproduction_max_samples`, `pca_components_search` and `mlflow_plotting` were not touched.
- The `vit_google` notebooks still pass already standardised inputs through the Hugging Face feature extractor, which applies its own fixed normalisation on top.
