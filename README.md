# vit-thesis

Code for my MSc thesis, *Vision Transformer for Gesture Recognition from EMG Signals* (MSc Digital Media & Computational Intelligence, Aristotle University of Thessaloniki): Vision Transformers (fine-tuned from a Google checkpoint, and trained from scratch) compared against SVM baselines across gesture-count and participant splits. The repository also holds a separate, independent pipeline for ViT-GPT2 image captioning on a spatial COCO dataset.

> TODO: one sentence on the research question and the headline result (author only).

**Status:** the package reimplements the original notebooks. Nothing has been re-run end to end on the real data since the refactor, so there are no verified results in this README (see [Results](#results) and [Known limitations](#known-limitations)).

## Repository layout

```
src/emg_gesture/   EMG package: data loading, splits, SVM, ViT training, evaluation, CLI (`emg-gesture`)
src/captioning/    captioning side package: `captioning subset|remove-empty|train|infer`
configs/           YAML configs: base.yaml < svm.yaml / vit_google.yaml / vit_scratch.yaml < task/<task>.yaml;
                   captioning.yaml for the captioning side
scripts/           prepare_folders.py: builds the per-task dataset folders
tests/             pytest suite (torch-dependent tests skip when torch is not installed)
03_experiments/    EMG experiment notebooks (PCA search, MLflow plotting, max-samples reproduction)
  archive/         original notebooks superseded by the package (see archive/README.md, archive/LEGACY_NOTES.md)
01_dataset/, 02_training/   captioning side work (not the EMG thesis); archive/ subfolders hold superseded scripts
requirements.txt, requirements-freeze.txt   pinned direct dependencies / full original environment freeze
```

Notebook suffixes mean the experimental split: `5gest` / `10gest` (5 or 10 gesture classes), `5part` / `10part` / `17participants` (number of participants).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[torch,dev]'
```

Extras: `torch` (torch, transformers; needed for `vit-google` / `vit-scratch`), `tracking` (mlflow), `dev` (pytest, ruff). The SVM baseline needs only the core install.

Alternatively `pip install -r requirements.txt` installs the thesis-era pinned set (Python 3.10). The original freeze used CUDA 11.8 builds (`+cu118`); for GPU install matching wheels from https://download.pytorch.org/whl/cu118 first.

**Verification caveat:** the tests were run against newer versions than the pins (torch 2.15.0 CPU, transformers 5.22.2, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, PyYAML 6.0.3). The pinned set in `requirements.txt` (torch 2.0.1, transformers 4.30.2, numpy 1.25.1, ...) has NOT been verified here.

## Configs and tasks

Tasks (`--task`), with configs in `configs/task/`:

| Task | Label | n_classes | Split behaviour |
|---|---|---|---|
| `10gest` | gesture | 10 | grouped by participant by default; `--split random` allowed |
| `5gest` | gesture | 5 | grouped by participant by default; `--split random` allowed (SVM kernel rbf) |
| `17participants` | participant | 17 | forced to `random` (logged) |
| `10part` | participant | 10 | forced to `random` (logged) |
| `5part` | participant | 5 | forced to `random` (logged) |

Participant-label tasks cannot be split by participant, hence the forced random per-file split (see limitations). Data layout: one flat folder of `*.mat` per task (e.g. `s1_s2_s3_matfiles_10gest/`), labels parsed from filenames.

## Run order

1. Create the task folders from the raw dataset (try `--dry-run` first):

   ```bash
   python scripts/prepare_folders.py --source-dir SRC --dest-dir DATA --task all --dry-run
   python scripts/prepare_folders.py --source-dir SRC --dest-dir DATA --task all
   ```

2. For each task `T` in `10gest 5gest 17participants 10part 5part`, run the three models:

   ```bash
   emg-gesture svm         --task T --seeds 1,2,3 --set data_dir=DATA/<task folder>
   emg-gesture vit-google  --task T --seeds 1,2,3 --set data_dir=DATA/<task folder>
   emg-gesture vit-scratch --task T --seeds 1,2,3 --set data_dir=DATA/<task folder>
   ```

   Other options: `--split grouped|random`, `--seed N` (single run; not combined with `--seeds`), `--config-dir`, `--out-dir` (default `runs`), repeatable `--set KEY=VALUE` (dotted keys, e.g. `svm.kernel=rbf`).

3. Optional: `03_experiments/mlflow_plotting.ipynb` for plots from tracked runs.

The captioning commands are independent (`captioning --help`) and take `--data-dir` (or `DATA_DIR`) and `--output-dir` (or `OUTPUT_DIR`).

## Outputs per run

Each run writes `runs/<task>_<method>_<seed>_<timestamp>/` containing:

- `config.resolved.yaml`: the merged config actually used
- `metrics.json`: test metrics, split mode (and why it was forced), stratification, per-split class counts, missing classes
- `confusion.png`, `per_class.csv`
- ViT runs only: `history.csv` (per-epoch curves) and `best_model.pth` (best validation checkpoint)

With `--seeds`, an extra `aggregate.json` (mean and sample std of test accuracy and macro-F1 across seeds) is written to its own `*-aggregate*` run directory.

## Reproducibility

- Seed: `seed` in `configs/base.yaml` (default 42), overridden by `--seed`. It seeds `random`, NumPy and torch/CUDA, and the DataLoader generator and worker init.
- Determinism: `--set deterministic=true` enables `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG`. Best effort only; results can still differ across hardware, torch versions and worker counts. Default is `false`.
- Report `--seeds 1,2,3,...` and the `aggregate.json` mean and std rather than a single run.
- The resolved config is saved with every run.

## Results

TODO: no results are claimed here. Fill in only after re-running everything with the current code.

| Model | Task | Split | Seeds | Test accuracy (mean +- std) | Macro-F1 |
|---|---|---|---|---|---|
| SVM | TODO | TODO | TODO | TODO | TODO |
| ViT (Google, fine-tuned) | TODO | TODO | TODO | TODO | TODO |
| ViT (scratch) | TODO | TODO | TODO | TODO | TODO |

## Data

TODO: where the data lives (not committed), how to obtain it, consent/licensing note (author only).

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
- Results are single runs by default; use `--seeds 1,2,3` (any command) to run several seeds and get an `aggregate.json` (mean and sample std of test accuracy and macro-F1), and report that rather than one test accuracy.
- ViT "channels" are image planes, not EMG channels: the flattened (time, channels) window is reshaped to 3x224x224 and normalised per plane, so the 3 planes mix time and EMG channels arbitrarily.
- A single grouped split uses few participants for validation/test, so results depend strongly on which participants land there; a class may be missing from a split (a warning is logged and `metrics.json` records `split_missing_classes` and per-split `class_counts`).
- Participant-label tasks (`*part*` tasks) are forced to a random per-file split, so train and test contain neighbouring windows of the same sessions; accuracy can be inflated by session adjacency and does not measure generalisation.
- The random split is stratified by label when every class has at least 2 members (config key `stratify`, top level because `split` is already the mode string); otherwise it falls back with a warning. `metrics.json` records the split mode, why a split was forced to random, and whether stratification was used.
- Scheduler (`vit_scratch`): the notebooks counted epochs as optimiser steps (`scheduler.step()` per batch with warmup/max given in epochs). The package reproduces that inside `max_epochs` steps but clamps beyond it (LR stays at `eta_min`), so long runs spend most epochs at the floor LR.
- Checkpoint selection uses validation accuracy with ties broken by lower validation loss.
- Determinism: `deterministic: true` (config) turns on `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG`. This is best effort only: ops without deterministic kernels just warn, and results can still differ across hardware, torch versions and worker counts.
- The legacy scratch cells that follow the main pipeline in `vit_train_from_scratch_5gest`, `_5part` and `_10part` were not updated (marked in a markdown cell); `folders_creation`, `code_reproduction_max_samples`, `pca_components_search` and `mlflow_plotting` were not touched.
- The `vit_google` notebooks still pass already standardised inputs through the Hugging Face feature extractor, which applies its own fixed normalisation on top.
