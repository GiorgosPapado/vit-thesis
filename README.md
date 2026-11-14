# vit-thesis

Code for my MSc thesis, *Vision Transformer for Gesture Recognition from EMG Signals* (MSc Digital Media & Computational Intelligence, Aristotle University of Thessaloniki): Vision Transformers (fine-tuned from a Google checkpoint, and trained from scratch) compared against SVM baselines across gesture-count and participant splits. The ViT-GPT2 captioning side work moved to a separate repository (`vit-gpt2-captioning`); the original history is in tag `pre-cleanup`.

**Research question and headline claim (from the thesis):** a comparative study of attention (Vision Transformer) against a traditional ML baseline (PCA + SVM) on two tasks, gesture recognition (HGR) and user identification (the thesis calls the latter "biometric authentication", but it is multi-class identification). The thesis reports superior performance of the ViT. It states no formal hypotheses. Numbers are summarised under [Results](#results) and were **not** reproduced by this repository.

**Status:** the package reimplements the original notebooks. Nothing has been re-run end to end on the real data since the refactor, so there are no verified results in this README (see [Results](#results) and [Known limitations](#known-limitations)).

## Repository layout

```
src/emg_gesture/   EMG package: data loading, splits, SVM, ViT training, evaluation, CLI (`emg-gesture`)
configs/           YAML configs: base.yaml < svm.yaml / vit_google.yaml / vit_scratch.yaml < task/<task>.yaml
scripts/           convert_dat_to_mat.py: GRABMyo WFDB (.dat/.hea) -> flat .mat folder; prepare_folders.py: builds the per-task dataset folders from it
tests/             pytest suite (torch-dependent tests skip when torch is not installed; CI requires them)
03_experiments/    remaining EMG notebooks (PCA search, MLflow plotting, max-samples reproduction); the other experiment notebooks were removed, see tag `pre-cleanup`
thesis-era-pins.txt   thesis-era pinned direct dependencies (EMG + torch only; unverified)
```

Task names mean the experimental split: `5gest` / `10gest` (5 or 10 gesture classes), `5part` / `10part` / `17participants` (number of participants).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,torch,tracking]"
```

Extras: `torch` (torch, transformers; **required** for the `vit-google` / `vit-scratch` commands), `convert` (wfdb; only for `scripts/convert_dat_to_mat.py`), `tracking` (mlflow; needed only for MLflow tracking and `mlflow_plotting.ipynb`), `dev` (pytest, ruff). The SVM baseline needs only the core install. Without the `torch` extra the ViT tests are skipped, not run.

**Supported Python and the two install paths**

- Path A, reproducible thesis-era set: `pip install -r thesis-era-pins.txt` (then `pip install -e .`). Targets **Python 3.10 or 3.11**; there are no Python 3.12 wheels for torch 2.0.1 / numpy 1.25.1. The original freeze (with CUDA 11.8 builds, `+cu118`) was in the deleted `requirements-freeze.txt`, which is available in tag `pre-cleanup`; for GPU install matching wheels from https://download.pytorch.org/whl/cu118 first. `wfdb` (for the converter) is listed unpinned in `thesis-era-pins.txt`.
- Path B, newer stack: `pip install -e ".[dev,torch,tracking]"` uses only lower bounds from `pyproject.toml` (`requires-python >=3.10`) and also works on newer Pythons and libraries (tests ran on Python 3.12 with torch 2.15, transformers 5.22).

`pyproject.toml` extras hold lower bounds; `thesis-era-pins.txt` holds the pinned reproducible set.

**Verification caveat:** the tests were run against newer versions than the pins (torch 2.15.0 CPU, transformers 5.22.2, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, PyYAML 6.0.3). The pinned set in `thesis-era-pins.txt` (torch 2.0.1, transformers 4.30.2, numpy 1.25.1, ...) has NOT been verified here. The `torch` extra deliberately has no upper cap.

## Configs and tasks

Tasks (`--task`), with configs in `configs/task/`:

| Task | Label | n_classes | Split behaviour |
|---|---|---|---|
| `10gest` | gesture | 10 | grouped by participant by default; `--split random` / `--split session` allowed |
| `5gest` | gesture | 5 | grouped by participant by default; `--split random` / `--split session` allowed (SVM kernel rbf) |
| `17participants` | gesture | 10 | grouped by participant by default; `--split random` / `--split session` allowed |
| `10part` | participant | 10 | forced to `random` (logged) |
| `5part` | participant | 5 | forced to `random` (logged) |

Participant-label tasks cannot be split by participant, hence the forced random per-file split (`runs.resolve_split`; this also overrides `--split session`, see limitations). Data layout: one flat folder of `*.mat` per task (e.g. `s1_s2_s3_matfiles_10gest/`), labels parsed from filenames. The CLI matches `gesture<k>`, `participant<k>` and `session<k>` strictly: a whole `_`-separated part of the file name must be exactly the token followed by digits (optionally with the extension), case-sensitive, so `session1_participant7_gesture10_trial1.mat` works but `mysession1_...` or `Session1_...` is not recognised; a file without the needed token raises an error.

**`17participants`:** resolved as gesture recognition (17 participants x the 10-gesture set, thesis Table 5.4), with the participant-disjoint grouped split by default. `scripts/prepare_folders.py` already selects the 10-gesture set with participant id <= 17.

**`.dat` to `.mat`:** the CLI reads `.mat` only, while GRABMyo is distributed as WFDB records (`.hea` + `.dat`). `scripts/convert_dat_to_mat.py` does the conversion (see Run order step 0). Not verified: the script was tested only on synthetic records, never on real GRABMyo files. Use `--flatten` so that all converted `.mat` land in one flat folder, which is what `scripts/prepare_folders.py` reads for every task (10gest selects the 10-gesture set from all sessions).

## Run order

0. Convert the raw GRABMyo WFDB records (`.hea` + `.dat`) to `.mat` (key `data_emg`, shape (time, channels)); needs the `convert` extra (`pip install -e ".[convert]"`):

   ```bash
   python scripts/convert_dat_to_mat.py --source-dir RAW --dest-dir SRC --dry-run
   python scripts/convert_dat_to_mat.py --source-dir RAW --dest-dir SRC --flatten [--channels 0,1,2] [--force]
   ```

   The base file name is kept; with `--flatten` all files go into `SRC` directly (two records with the same base name are an error), without it the relative folder layout is mirrored (but `prepare_folders.py` needs the flat layout). Channel indices must be unique and within the channel count, all records must have the same channel count, NaNs are reported per record, existing `.mat` files are not overwritten without `--force`, and `SRC/manifest.json` records fs, units, sig_name, channels and the number of records. So the `session<k>`, `participant<k>`, `gesture<k>`, `trial<k>` tokens survive. Assumption: the raw records are named like `session1_participant1_gesture1_trial1` (the naming `scripts/prepare_folders.py` and the CLI expect); if the raw tree names differ, rename the files or adjust before step 1. Files without a `.hea` header are skipped. `--channels` takes 0-based indices; without it all channels of the record are kept (the thesis used 28 of 32, which this script does not select for you). Not run on real GRABMyo files.

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

   Optional extra baselines and options (added after the thesis; **not run on real data**, no results exist for them):

   - `emg-gesture features --task T`: handcrafted per-channel features (RMS, MAV, WL, ZC, SSC, var) -> StandardScaler -> `lda` | `rf` | `svm` (`features.classifier`, see `configs/features.yaml`).
   - `--set svm.grid_search=true` (default `false`): GridSearchCV over `svm.grid` (C, kernel, gamma) using the training files only (train-only CV); Kernel `linear` is searched without `gamma` (no redundant candidates). `metrics.json` records `best_params`, `grid`, `best_cv_score` and `cv_is_selection_biased: true`; `cv_scores`/`cv_mean` are omitted for grid runs because the best CV score is selection-biased. The held-out test set is the unbiased estimate (fixed runs record `cv_scores`, `cv_mean`, `cv_is_selection_biased: false`). The PCA size stays fixed.
   - `--set model.input_repr=spectrogram` (ViT; default `flatten`): STFT spectrogram image of the full window (`time_covered: 1.0`, `truncated: false`; `retained_fraction` is null because information retained is not comparable to the flatten path). The 3 image planes are averages of 3 contiguous channel groups, so channel identity is not preserved; `vit_input` records the effective `nperseg`/`noverlap` and frequency-bin count. Augmentation is applied to the normalised image and confounds flatten-vs-spectrogram comparisons unless disabled. Experimental and untested on real data.

   Other options: `--split grouped|random|session`, `--seed N` (single run; not combined with `--seeds`), `--config-dir`, `--out-dir` (default `runs`), repeatable `--set KEY=VALUE` (dotted keys, e.g. `svm.kernel=rbf`). Scalar grid values such as `--set svm.grid.C=1` are wrapped into a one-element list. The default `--config-dir` is the relative `configs/`, so run from the repository root or pass `--config-dir`.

3. Optional: `03_experiments/mlflow_plotting.ipynb` for plots from tracked runs.

## Cross-session split (closest to the thesis protocol)

The thesis split is cross-session: sessions 1 and 2 train, session 3 tests, the same split for SVM and ViT. Reproduce that split with:

```bash
emg-gesture svm --task 10gest --split session --set data_dir=DATA/<folder with all three sessions>
```

**Required folder layout:** one flat folder of `.mat` files from sessions 1, 2 and 3, each file name containing a `session<k>` token (`session1_`, `session2_`, `session3_`, e.g. `session2_participant7_gesture10_trial1.mat`), with train and test sessions both present. The `10gest` and `5gest` / `17participants` folders from `scripts/prepare_folders.py` keep all sessions and so provide this; for `10part` / `5part` only `session3_*` files, so `--split session` on such a folder raises `ValueError` (no files for the train or test sessions). Use the 10gest folder (or assemble one yourself).

File names need a strict `session<k>` token (a whole `_`-separated part such as `session1`; see Configs and tasks); a file without one raises `ValueError`. Config keys `train_sessions` (default `[1, 2]`) and `test_sessions` (default `[3]`) select the sessions; files from other sessions are excluded, and the script asserts that no file is in two splits. `test_size` is ignored. The thesis described no validation set, so for the ViT a validation set is carved out of the TRAIN sessions as a random per-file fraction (`val_size`, stratified when possible, not participant-disjoint); the SVM uses none (it trains on all train-session files, records no `val_split` other than `none`, and ignores `val_sessions`). This random per-file validation split puts sibling repetitions of the same recording in train and validation, so validation accuracy is optimistic and checkpoint selection (best validation accuracy) is weak; for ViT runs a warning is logged and `metrics.json` records `val_split: random_per_file`. Setting `val_sessions` (e.g. `--set val_sessions=[2]`, a proper subset of `train_sessions`) holds out whole sessions for validation instead (`val_split: session_holdout` and `val_sessions` in `metrics.json`; `val_size` is then ignored); it is opt-in, applies to ViT runs only, and is untested on real data. `metrics.json` records `split_mode`, `generalises_to_new_subjects` (false for session), `val_split`, `val_sessions`, `train_sessions` and `test_sessions` (the session lists are null for the other modes). With `--seeds`, `aggregate.json` also records `split_mode`, `generalises_to_new_subjects`, `train_sessions` and `test_sessions`; in session mode the train and test files are fixed, so the std over seeds reflects model initialisation and the validation split only. Participant-label tasks (`10part`, `5part`) are still forced to a random per-file split even with `--split session` (the reason is recorded as `split_forced_reason`), so this mode is only meaningful for `10gest` and `5gest`.

This is the closest match to the thesis protocol but still **not a verified reproduction**: the thesis input pipeline (scalograms, PCA input shape) is not fully specified, and the repo's preprocessing differs (see limitations).

## Tests and lint

```bash
python -m pytest -q -rs     # or: make test
python -m ruff check .      # CI; `make lint` checks src tests scripts only
REQUIRE_TORCH=1 python -m pytest -q -rs -p no:cacheprovider   # or: make test-all (fails if torch is missing)
```

Ruff is configured in `pyproject.toml` (line length 120; rules E, F, I, W) and excludes `*.ipynb`. CI (`.github/workflows/ci.yml`) runs ruff and pytest on Python 3.10 without torch, plus a **required** job with CPU torch that runs the suite with `REQUIRE_TORCH=1` (torch-dependent tests must run, not skip). `tests/test_e2e_smoke.py` is an end-to-end smoke test that runs each CLI command on synthetic data and checks the artifacts; it does not check accuracy.

## Outputs per run

Each run writes `runs/<task>_<method>_<seed>_<timestamp>/` containing:

- `config.resolved.yaml`: the merged config actually used
- `metrics.json`: test metrics, `split` / `split_mode` (and why it was forced), `generalises_to_new_subjects` (true only for the grouped split), `val_split` / `val_sessions`, `train_sessions` / `test_sessions` (session mode), stratification, per-split class counts, missing classes; SVM runs record `svm_hyperparameters` (`fixed (no grid search)` or `grid_search`), `svm_config` (fixed runs: the SVM config; grid runs: only `pca_components` and `grid`, the tuned values are in `best_params`) and `svm_uses_full_window`; grid runs also record `best_params`, `grid`, `best_cv_score` and `cv_is_selection_biased: true`; `features` runs record `feature_names`, `classifier` and `classifier_params`; ViT runs record `vit_input` (flatten: window shape, `retained_fraction`, `retained_time_samples`, truncation; spectrogram: `retained_fraction` null, `input_repr`, `time_covered`, `truncated: false`, effective `nperseg` / `noverlap`, `n_frequency_bins`, `channel_grouping`)
- `confusion.png`, `per_class.csv`
- ViT runs only: `history.csv` (per-epoch curves) and `best_model.pth` (best validation checkpoint, a `state_dict` reloaded for the single test evaluation with `torch.load(..., weights_only=True)`)

With `--seeds`, an extra `aggregate.json` is written to its own `*-aggregate*` run directory. It holds `task`, `method`, `seeds`, `n_runs`, `per_seed` (seed, run directory, test accuracy, macro-F1), `std_ddof`, `test_accuracy` and `macro_f1` (each `mean` and sample `std`), plus `split_mode`, `generalises_to_new_subjects`, `train_sessions` and `test_sessions` copied from the first run; in session mode it also has a `std_note`.

## Reproducibility

- Seed: `seed` in `configs/base.yaml` (default 42), overridden by `--seed`. It seeds `random`, NumPy and torch/CUDA, and the DataLoader generator and worker init.
- Determinism: `--set deterministic=true` enables `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG`. Best effort only; results can still differ across hardware, torch versions and worker counts. Default is `false`.
- Report `--seeds 1,2,3,...` and the `aggregate.json` mean and std rather than a single run.
- The resolved config is saved with every run.

## Results

**Reported in the thesis (cross-session protocol), not reproduced by this repository.** Metric: accuracy only, single run, no seeds, sessions 1+2 train / session 3 test (same split for SVM and ViT); no separate validation set is described. The original notebooks used random per-file splits (and the `emg_gesture` package defaults to a participant-grouped split), so this repository does NOT reproduce the thesis protocol and its numbers will differ. "Pre-trained" = `vit_google`, "scratch" = `vit_train_from_scratch` (removed notebooks, see tag `pre-cleanup`).

Table 5.2, user identification (tasks `5part`, `10part`; removed notebooks, see tag `pre-cleanup`):

| Participants (samples) | SVM | ViT pre-trained | ViT scratch |
|---|---|---|---|
| 5 (1785) | 0.4761 | 0.7619 | 0.6078 |
| 10 (3570) | 0.2619 | 0.6171 | 0.4165 |

Table 5.3, gesture recognition, 43 participants (tasks `5gest`, `10gest`; removed notebooks, see tag `pre-cleanup`):

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
- Input to the ViT: the thesis describes the EMG as rendered to a 224x224 image by averaging channel amplitudes; the exact transform is not described (unclear). This repository's code does NOT do that (default `flatten`; `spectrogram` differs, see Run order): it flattens the (time, channels) window row-major, pads or truncates it to 3*224*224 values and reshapes it to 3x224x224 (see [Known limitations](#known-limitations)).
- Software and hardware: PyTorch + MLflow, Python 3.10.12, Ubuntu 22.04; Intel i9-10900X, 2x RTX 3090, 64 GB RAM.

## Data

Dataset: GRABMyo, Pradhan, Jiang, He, "Gesture Recognition and Biometrics ElectroMyogram (GRABMyo) Dataset", IEEE Dataport, 2022; also published as Pradhan, He, Jiang (2022), Scientific Data 9, 733. It is not committed to this repository.

- 43 participants, 3 sessions; 32 channels (16 forearm + 12 wrist, the other 4 unused), 2048 Hz.
- 17 classes: 16 dynamic gestures + rest; 7 repetitions per gesture per session.
- Each sample is 5 s = 10240 samples x 32 channels; 15351 samples in total.
- Inconsistency in the source: the thesis text also says "20 male and 20 female", which does not sum to 43.

Obtaining the data and licence:

- The dataset is on PhysioNet: https://physionet.org/content/grabmyo/
- Licence: Creative Commons Attribution 4.0, per the PhysioNet / AWS Open Data listing. Confirm the exact licence version on the dataset page before redistributing anything.
- It can be downloaded without an account: `aws s3 sync --no-sign-request s3://physionet-open/grabmyo/ <dir>` (AWS registry: https://registry.opendata.aws/grabmyo/).
- These facts were taken from web listings and were not verified against the downloaded files.

## Citation

```
Papadopoulos, G. (2024). Vision Transformer for Gesture Recognition from EMG Signals.
MSc thesis, Aristotle University of Thessaloniki, School of Informatics,
MSc Digital Media - Computational Intelligence.

Pradhan, A., Jiang, N., He, J. (2022). Gesture Recognition and Biometrics ElectroMyogram
(GRABMyo) Dataset. IEEE Dataport.

Pradhan, A., He, J., Jiang, N. (2022). Multi-day dataset of forearm and wrist
electromyogram for hand gesture recognition and biometrics. Scientific Data 9, 733.

Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale. arXiv:2010.11929.
```

## License

MIT, see [LICENSE](LICENSE).

## Known limitations

These points come from a code review; they are not experimental results.

**Earlier notebook-era numbers (thesis text, slides, earlier READMEs) came from older code with a different pipeline and are NOT comparable with this package. The old notebooks were removed from the tree; the full history is tagged `pre-cleanup`. Re-run everything before quoting a number.** The `features` command, `svm.grid_search` and `model.input_repr: spectrogram` have not been run on real data.

- In `*gest*` mode `'random'` the numbers are within-subject and do not measure generalisation to new users.
- Results are single runs by default; use `--seeds 1,2,3` (any command) to run several seeds and get an `aggregate.json` (mean and sample std of test accuracy and macro-F1), and report that rather than one test accuracy.
- ViT input truncation: `load_sample` flattens the (target_length, channels) window and keeps only the first 3*224*224 = 150528 values. For a window with more values than that, the ViT sees only the leading part in time (that is `150528 / channels` time samples), while the SVM (`load_flattened`) sees the full window. Not an averaging of channels, and not a like-for-like comparison of inputs. A warning is logged, and `metrics.json` records `vit_input` (window shape, `retained_fraction`, `retained_time_samples`, `truncated`) and that the SVM uses the full window. The behaviour itself was not changed (default `flatten`; the opt-in `spectrogram` input covers the full window).
- SVM baseline: this package runs a fixed-hyperparameter pipeline (channel standardisation, StandardScaler, PCA, SVC; defaults `kernel: linear`, `C: 1`, `pca_components: 100`, per-task overrides in `configs/`) with no grid search; the thesis SVM (PCA(100) + rbf with GridSearchCV, see above) is therefore not reproduced. `metrics.json` records `svm_hyperparameters: fixed (no grid search)`. Opt-in `svm.grid_search` (train-only CV) exists but has not been run on real data.
- Split generalisation: only `grouped` keeps participants disjoint (`generalises_to_new_subjects: true` in `metrics.json` / `aggregate.json`). `random` has no group key, so the same participant and session can be in train and test. `session` trains and tests on the same participants (within-subject, cross-session), so it does not measure generalisation to new users. Its validation split is random per file unless the opt-in `val_sessions` is set (see [Cross-session split](#cross-session-split-closest-to-the-thesis-protocol)).
- ViT "channels" are image planes, not EMG channels (default `flatten`; with `input_repr: spectrogram` each plane is the average of a contiguous channel group instead): the flattened (time, channels) window is reshaped to 3x224x224 and normalised per plane, so the 3 planes mix time and EMG channels arbitrarily.
- A single grouped split uses few participants for validation/test, so results depend strongly on which participants land there; a class may be missing from a split (a warning is logged and `metrics.json` records `split_missing_classes` and per-split `class_counts`).
- Participant-label tasks (`10part`, `5part`) are forced to a random per-file split, so train and test contain neighbouring windows of the same sessions; accuracy can be inflated by session adjacency and does not measure generalisation.
- The random split is stratified by label when every class has at least 2 members (config key `stratify`, top level because `split` is already the mode string); otherwise it falls back with a warning. `metrics.json` records the split mode, why a split was forced to random, and whether stratification was used.
- Scheduler (`vit_scratch`): the original notebooks counted epochs as optimiser steps (`scheduler.step()` per batch with warmup/max given in epochs). The package reproduces that inside `max_epochs` steps but clamps beyond it (LR stays at `eta_min`), so long runs spend most epochs at the floor LR.
- Features/SVM zero-pad recordings shorter than `target_length`, so recording length leaks into the amplitude features (RMS, MAV, var) and the SVM input; ZC and SSC count sign / slope changes with no amplitude threshold, so noise-level jitter around zero is counted.
- Grid-search `test_accuracy` is a single train/test split with no outer loop (the CV inside is train-only and selection-biased), so it is one noisy estimate; use `--seeds`.
- Checkpoint selection uses validation accuracy with ties broken by lower validation loss.
- Determinism: `deterministic: true` (config) turns on `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG`. This is best effort only: ops without deterministic kernels just warn, and results can still differ across hardware, torch versions and worker counts.
- Protocol difference: the thesis used a cross-session split (sessions 1+2 train, session 3 test; single run, accuracy only). The original notebooks used random per-file splits and the package defaults to participant-grouped, so neither reproduces the thesis numbers; `--split session` is the closest match (see [Cross-session split](#cross-session-split-closest-to-the-thesis-protocol)).
