# vit-thesis

Code for my MSc thesis, *Vision Transformer for Gesture Recognition from EMG Signals* (MSc Digital Media & Computational Intelligence, Aristotle University of Thessaloniki): Vision Transformers (fine-tuned from a Google checkpoint, and trained from scratch) compared against SVM baselines across gesture-count and participant splits. The repository also holds a separate, independent pipeline for ViT-GPT2 image captioning on a spatial COCO dataset.

**Research question and headline claim (from the thesis):** a comparative study of attention (Vision Transformer) against a traditional ML baseline (PCA + SVM) on two tasks, gesture recognition (HGR) and user identification (the thesis calls the latter "biometric authentication", but it is multi-class identification). The thesis reports superior performance of the ViT. It states no formal hypotheses. Numbers are summarised under [Results](#results) and were **not** reproduced by this repository.

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
pip install -e ".[dev,torch,tracking]"
```

Extras: `torch` (torch, transformers; **required** for the `vit-google` / `vit-scratch` commands), `tracking` (mlflow; needed only for MLflow tracking and `mlflow_plotting.ipynb`), `dev` (pytest, ruff). The SVM baseline needs only the core install. Without the `torch` extra the ViT tests are skipped, not run.

**Supported Python and the two install paths**

- Path A, reproducible thesis-era set: `pip install -r requirements.txt` (then `pip install -e .`). Targets **Python 3.10 or 3.11**; there are no Python 3.12 wheels for torch 2.0.1 / numpy 1.25.1. The original freeze used CUDA 11.8 builds (`+cu118`); for GPU install matching wheels from https://download.pytorch.org/whl/cu118 first.
- Path B, newer stack: `pip install -e ".[dev,torch,tracking]"` uses only lower bounds from `pyproject.toml` (`requires-python >=3.10`) and also works on newer Pythons and libraries (tests ran on Python 3.12 with torch 2.15, transformers 5.22).

`pyproject.toml` extras hold lower bounds; `requirements.txt` holds the pinned reproducible set.

**Verification caveat:** the tests were run against newer versions than the pins (torch 2.15.0 CPU, transformers 5.22.2, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, PyYAML 6.0.3). The pinned set in `requirements.txt` (torch 2.0.1, transformers 4.30.2, numpy 1.25.1, ...) has NOT been verified here. In particular, the captioning trainer is verified in tests only against transformers 5; the pinned transformers 4.30.2 is unverified for captioning (the `torch` extra deliberately has no upper cap).

## Configs and tasks

Tasks (`--task`), with configs in `configs/task/`:

| Task | Label | n_classes | Split behaviour |
|---|---|---|---|
| `10gest` | gesture | 10 | grouped by participant by default; `--split random` / `--split session` allowed |
| `5gest` | gesture | 5 | grouped by participant by default; `--split random` / `--split session` allowed (SVM kernel rbf) |
| `17participants` | participant | 17 | forced to `random` (logged) |
| `10part` | participant | 10 | forced to `random` (logged) |
| `5part` | participant | 5 | forced to `random` (logged) |

Participant-label tasks cannot be split by participant, hence the forced random per-file split (`runs.resolve_split`; this also overrides `--split session`, see limitations). Data layout: one flat folder of `*.mat` per task (e.g. `s1_s2_s3_matfiles_10gest/`), labels parsed from filenames. The CLI matches `gesture<k>`, `participant<k>` and `session<k>` strictly: a whole `_`-separated part of the file name must be exactly the token followed by digits (optionally with the extension), case-sensitive, so `session1_participant7_gesture10_trial1.mat` works but `mysession1_...` or `Session1_...` is not recognised; a file without the needed token raises an error.

**`17participants` label discrepancy (unresolved):** `configs/task/17participants.yaml` sets `label: participant` (so the task is forced to a random split and is identification), whereas the thesis Table 5.4 and the archived `svm_17participants` notebook describe it as gesture recognition (label = gesture). Check this before comparing with the thesis table. TODO(author): decide which is intended; the code was not changed.

**10gest data gap:** `scripts/prepare_folders.py --task 10gest` copies per-participant `.dat` files, whereas the CLI reads `.mat` only. The `.dat` to `.mat` conversion happened outside this repo, so `10gest` runs need the `.mat` files produced by the author's conversion step. TODO(author): state where that conversion lives (script/notebook/tool) and add or link it.

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

   Other options: `--split grouped|random|session`, `--seed N` (single run; not combined with `--seeds`), `--config-dir`, `--out-dir` (default `runs`), repeatable `--set KEY=VALUE` (dotted keys, e.g. `svm.kernel=rbf`).

3. Optional: `03_experiments/mlflow_plotting.ipynb` for plots from tracked runs.

The captioning commands are independent (`captioning --help`) and take `--data-dir` (or `DATA_DIR`) and `--output-dir` (or `OUTPUT_DIR`).

## Cross-session split (closest to the thesis protocol)

The thesis split is cross-session: sessions 1 and 2 train, session 3 tests, the same split for SVM and ViT. Reproduce that split with:

```bash
emg-gesture svm --task 10gest --split session --set data_dir=DATA/<folder with all three sessions>
```

**Required folder layout:** one flat folder of `.mat` files from sessions 1, 2 and 3, each file name containing a `session<k>` token (`session1_`, `session2_`, `session3_`, e.g. `session2_participant7_gesture10_trial1.mat`), with train and test sessions both present. The output of `scripts/prepare_folders.py` does NOT provide this: for `10gest` it copies only `session1_*` `.dat` files, and for `10part` / `5part` only `session3_*` files, so `--split session` on such a folder raises `ValueError` (no files for the train or test sessions). You must assemble a suitable folder yourself.

File names need a strict `session<k>` token (a whole `_`-separated part such as `session1`; see Configs and tasks); a file without one raises `ValueError`. Config keys `train_sessions` (default `[1, 2]`) and `test_sessions` (default `[3]`) select the sessions; files from other sessions are excluded, and the script asserts that no file is in two splits. `test_size` is ignored. The thesis described no validation set, so for the ViT a validation set is carved out of the TRAIN sessions as a random per-file fraction (`val_size`, stratified when possible, not participant-disjoint); the SVM uses none (it trains on all train-session files, records no `val_split` other than `none`, and ignores `val_sessions`). This random per-file validation split puts sibling repetitions of the same recording in train and validation, so validation accuracy is optimistic and checkpoint selection (best validation accuracy) is weak; for ViT runs a warning is logged and `metrics.json` records `val_split: random_per_file`. Setting `val_sessions` (e.g. `--set val_sessions=[2]`, a proper subset of `train_sessions`) holds out whole sessions for validation instead (`val_split: session_holdout` and `val_sessions` in `metrics.json`; `val_size` is then ignored); it is opt-in, applies to ViT runs only, and is untested on real data. `metrics.json` records `split_mode`, `generalises_to_new_subjects` (false for session), `val_split`, `val_sessions`, `train_sessions` and `test_sessions` (the session lists are null for the other modes). With `--seeds`, `aggregate.json` also records `split_mode`, `generalises_to_new_subjects`, `train_sessions` and `test_sessions`; in session mode the train and test files are fixed, so the std over seeds reflects model initialisation and the validation split only. Participant-label tasks (`*part*`, `17participants`) are still forced to a random per-file split even with `--split session` (the reason is recorded as `split_forced_reason`), so this mode is only meaningful for `10gest` and `5gest`.

This is the closest match to the thesis protocol but still **not a verified reproduction**: the thesis input pipeline (scalograms, PCA input shape) is not fully specified, and the repo's preprocessing differs (see limitations).

## Tests and lint

```bash
python -m pytest -q -rs     # or: make test
python -m ruff check .      # CI; `make lint` checks src tests scripts only
REQUIRE_TORCH=1 python -m pytest -q -rs -p no:cacheprovider   # or: make test-all (fails if torch is missing)
```

Ruff is configured in `pyproject.toml` (line length 120; rules E, F, I, W) and excludes `*/archive`, `*.ipynb` and `01_dataset/tester.py`, i.e. the archived notebooks/scripts and the legacy captioning script are not linted. CI (`.github/workflows/ci.yml`) runs ruff and pytest on Python 3.10 without torch, plus an allowed-to-fail job with CPU torch.

## Outputs per run

Each run writes `runs/<task>_<method>_<seed>_<timestamp>/` containing:

- `config.resolved.yaml`: the merged config actually used
- `metrics.json`: test metrics, `split` / `split_mode` (and why it was forced), `generalises_to_new_subjects` (true only for the grouped split), `val_split` / `val_sessions`, `train_sessions` / `test_sessions` (session mode), stratification, per-split class counts, missing classes; SVM runs record `svm_hyperparameters` (fixed, no grid search) and `svm_uses_full_window`; ViT runs record `vit_input` (window shape, retained fraction of the window, truncation)
- `confusion.png`, `per_class.csv`
- ViT runs only: `history.csv` (per-epoch curves) and `best_model.pth` (best validation checkpoint, a `state_dict` reloaded for the single test evaluation with `torch.load(..., weights_only=True)`)

With `--seeds`, an extra `aggregate.json` is written to its own `*-aggregate*` run directory. It holds `task`, `method`, `seeds`, `n_runs`, `per_seed` (seed, run directory, test accuracy, macro-F1), `std_ddof`, `test_accuracy` and `macro_f1` (each `mean` and sample `std`), plus `split_mode`, `generalises_to_new_subjects`, `train_sessions` and `test_sessions` copied from the first run; in session mode it also has a `std_note`.

## Reproducibility

- Seed: `seed` in `configs/base.yaml` (default 42), overridden by `--seed`. It seeds `random`, NumPy and torch/CUDA, and the DataLoader generator and worker init.
- Determinism: `--set deterministic=true` enables `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG`. Best effort only; results can still differ across hardware, torch versions and worker counts. Default is `false`.
- Report `--seeds 1,2,3,...` and the `aggregate.json` mean and std rather than a single run.
- The resolved config is saved with every run.

## Results

**Reported in the thesis (cross-session protocol), not reproduced by this repository.** Metric: accuracy only, single run, no seeds, sessions 1+2 train / session 3 test (same split for SVM and ViT); no separate validation set is described. The notebooks here use random per-file splits (and the `emg_gesture` package defaults to a participant-grouped split), so this repository does NOT reproduce the thesis protocol and its numbers will differ. "Pre-trained" = `vit_google`, "scratch" = `vit_train_from_scratch`.

Table 5.2, user identification (notebooks `5part`, `10part`):

| Participants (samples) | SVM | ViT pre-trained | ViT scratch |
|---|---|---|---|
| 5 (1785) | 0.4761 | 0.7619 | 0.6078 |
| 10 (3570) | 0.2619 | 0.6171 | 0.4165 |

Table 5.3, gesture recognition, 43 participants (notebooks `5gest`, `10gest`):

| Gestures (samples) | SVM | ViT pre-trained | ViT scratch |
|---|---|---|---|
| 5 (4515) | 0.6223 | 0.8315 | 0.7583 |
| 10 (9030) | 0.4672 | 0.6900 | 0.6039 |

Table 5.4, same sample size (3570 samples, 10 classes):

| Task | SVM | ViT pre-trained | ViT scratch |
|---|---|---|---|
| Identification, 10 participants (`10part`) | 0.2619 | 0.6171 | 0.4165 |
| Gesture recognition, 17 participants x 10 gestures (`17participants`) | 0.4039 | 0.6269 | 0.5287 |

Table 5.1, average EER: 0.033 (GRABMyo paper) vs 0.041 (thesis reproduction on the 9030-sample HGR partition).

**Thesis conclusions:** the ViT beat the SVM in all tasks; the pre-trained ViT plateaus after about a dozen epochs and is prone to overfitting; the scratch ViT scales better with more data; identification is harder than gesture recognition and both models fail to generalise well there.

**Thesis limitations / future work:** small subsets, high dimensionality, memory cost; future work lists FFT/wavelet preprocessing, data augmentation, and inference time / real-time use.

Results of this repository's own code: none yet. TODO: re-run with the current code and report mean +- std over seeds here.

## Method and setup (as described in the thesis)

- SVM: PCA(100) + SVC, rbf kernel, GridSearchCV over C in {0.1, 1, 10, 100} and gamma in {scale, auto, 0.01, 0.001, 0.0001}; best C=1, gamma=scale.
- ViT pre-trained: Google ViT checkpoint (variant not named in the thesis), 50 epochs.
- ViT from scratch: hidden size 128, 4 heads, 4 layers, 224x224 input, reduced patch count, higher dropout, 500 epochs.
- Training: AdamW, cross-entropy, 5 warm-up epochs, learning rate 1e-4, batch size 32.
- Input to the ViT: the thesis describes the EMG as rendered to a 224x224 image by averaging channel amplitudes; the exact transform is not described (unclear). This repository's code does NOT do that: it flattens the (time, channels) window row-major, pads or truncates it to 3*224*224 values and reshapes it to 3x224x224 (see [Known limitations](#known-limitations)).
- Software and hardware: PyTorch + MLflow, Python 3.10.12, Ubuntu 22.04; Intel i9-10900X, 2x RTX 3090, 64 GB RAM.

## Data

Dataset: GRABMyo, Pradhan, Jiang, He, "Gesture Recognition and Biometrics ElectroMyogram (GRABMyo) Dataset", IEEE Dataport, 2022. It is not committed to this repository.

- 43 participants, 3 sessions; 32 channels (16 forearm + 12 wrist, the other 4 unused), 2048 Hz.
- 17 classes: 16 dynamic gestures + rest; 7 repetitions per gesture per session.
- Each sample is 5 s = 10240 samples x 32 channels; 15351 samples in total.
- Inconsistency in the source: the thesis text also says "20 male and 20 female", which does not sum to 43.

TODO(author): dataset licence/terms (not stated in the thesis) and how to obtain the data.

## Citation

```
Papadopoulos, G. (2024). Vision Transformer for Gesture Recognition from EMG Signals.
MSc thesis, Aristotle University of Thessaloniki, School of Informatics,
MSc Digital Media - Computational Intelligence.

Pradhan, A., Jiang, N., He, J. (2022). Gesture Recognition and Biometrics ElectroMyogram
(GRABMyo) Dataset. IEEE Dataport.

Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale. arXiv:2010.11929.
```

## License

MIT, see [LICENSE](LICENSE).

## Known limitations

These points come from a code review of the notebooks; they are not experimental results.

**Important: the archived notebooks were edited (seed cells, pipeline changes, `SPLIT_MODE`, outputs cleared) and have NOT been re-run; they are not unchanged originals. Any accuracy numbers published or reported from earlier runs (thesis text, slides, earlier READMEs) used the old, flawed code and are NOT comparable with what the current notebooks will produce. Re-run everything before quoting a number.**

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
- ViT input truncation: `load_sample` flattens the (target_length, channels) window and keeps only the first 3*224*224 = 150528 values. For a window with more values than that, the ViT sees only the leading part in time (that is `150528 / channels` time samples), while the SVM (`load_flattened`) sees the full window. Not an averaging of channels, and not a like-for-like comparison of inputs. A warning is logged, and `metrics.json` records `vit_input` (window shape, `retained_fraction`, `retained_time_samples`, `truncated`) and that the SVM uses the full window. The behaviour itself was not changed.
- SVM baseline: this package runs a fixed-hyperparameter pipeline (channel standardisation, StandardScaler, PCA, SVC; defaults `kernel: linear`, `C: 1`, `pca_components: 100`, per-task overrides in `configs/`) with no grid search; the thesis SVM (PCA(100) + rbf with GridSearchCV, see above) is therefore not reproduced. `metrics.json` records `svm_hyperparameters: fixed (no grid search)`.
- Split generalisation: only `grouped` keeps participants disjoint (`generalises_to_new_subjects: true` in `metrics.json` / `aggregate.json`). `random` has no group key, so the same participant and session can be in train and test. `session` trains and tests on the same participants (within-subject, cross-session), so it does not measure generalisation to new users. Its validation split is random per file unless the opt-in `val_sessions` is set (see [Cross-session split](#cross-session-split-closest-to-the-thesis-protocol)).
- ViT "channels" are image planes, not EMG channels: the flattened (time, channels) window is reshaped to 3x224x224 and normalised per plane, so the 3 planes mix time and EMG channels arbitrarily.
- A single grouped split uses few participants for validation/test, so results depend strongly on which participants land there; a class may be missing from a split (a warning is logged and `metrics.json` records `split_missing_classes` and per-split `class_counts`).
- Participant-label tasks (`*part*` tasks) are forced to a random per-file split, so train and test contain neighbouring windows of the same sessions; accuracy can be inflated by session adjacency and does not measure generalisation.
- The random split is stratified by label when every class has at least 2 members (config key `stratify`, top level because `split` is already the mode string); otherwise it falls back with a warning. `metrics.json` records the split mode, why a split was forced to random, and whether stratification was used.
- Scheduler (`vit_scratch`): the notebooks counted epochs as optimiser steps (`scheduler.step()` per batch with warmup/max given in epochs). The package reproduces that inside `max_epochs` steps but clamps beyond it (LR stays at `eta_min`), so long runs spend most epochs at the floor LR.
- Checkpoint selection uses validation accuracy with ties broken by lower validation loss.
- Determinism: `deterministic: true` (config) turns on `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG`. This is best effort only: ops without deterministic kernels just warn, and results can still differ across hardware, torch versions and worker counts.
- The legacy scratch cells that follow the main pipeline in `vit_train_from_scratch_5gest`, `_5part` and `_10part` were not updated (marked in a markdown cell); `folders_creation`, `code_reproduction_max_samples`, `pca_components_search` and `mlflow_plotting` were not touched.
- The `vit_google` notebooks still pass already standardised inputs through the Hugging Face feature extractor, which applies its own fixed normalisation on top.
- Protocol difference: the thesis used a cross-session split (sessions 1+2 train, session 3 test; single run, accuracy only). The notebooks use random per-file splits and the package defaults to participant-grouped, so neither reproduces the thesis numbers; `--split session` is the closest match (see [Cross-session split](#cross-session-split-closest-to-the-thesis-protocol)).
