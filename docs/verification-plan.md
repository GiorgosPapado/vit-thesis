# Verification plan

Everything the repository still cannot claim, and what it takes to close each gap. Status of each item is
**not done** until the acceptance check passes. Do not add accuracy numbers to the README before Part B.

## Part A: can be done without the real data (no GPU, no account)

| # | Task | Files | Acceptance check |
|---|---|---|---|
| A1 | Include `wfdb` in the `REQUIRE_TORCH` guard so the converter tests cannot skip silently in the torch CI job | `tests/conftest.py` | With `wfdb` uninstalled and `REQUIRE_TORCH=1`, `pytest` exits with a usage error naming `wfdb`. With it installed, the suite passes with 0 skips. |
| A2 | Cache the spectrogram per file (or vectorise `_resize_bilinear`) so epochs do not recompute the STFT and the resize | `src/emg_gesture/data.py`, `src/emg_gesture/train_vit.py` | The existing spectrogram tests pass unchanged. A new test asserts the cached and uncached images are identical (`np.array_equal`). A timing check on 28 channels × 10240 samples shows a faster second epoch. |
| A3 | Add `.gitattributes` with `* text=auto eol=lf` | `.gitattributes` | `git status` shows no LF/CRLF warnings after a fresh checkout on Windows. |
| A4 | Replace the private `terminalreporter._session.exitstatus` hook with a supported mechanism (for example `pytest_sessionfinish` setting `session.exitstatus`) | `tests/conftest.py` | `REQUIRE_TORCH=1` still fails the run when a test is skipped, and passes when none is. |
| A5 | Script that turns `aggregate.json` files into a Markdown results table | `scripts/make_results_table.py`, `tests/test_make_results_table.py` | The test builds synthetic aggregates for 3 methods × 2 tasks and checks the table has mean ± std, `n_runs`, split mode and the `generalises_to_new_subjects` flag. The script refuses to mix aggregates with different split modes. |
| A6 | Confirm the licence and S3 path on the PhysioNet page for the version used | `README.md` (Data section) | The README cites the exact version, DOI and licence text copied from that page. The "taken from web listings" caveat is removed. (Manual, a few minutes.) |

## Part B: needs the real data (and a GPU for the ViT runs)

### B1. Data path
1. Download GRABMyo: `aws s3 sync --no-sign-request s3://physionet-open/grabmyo/ <dir>` (about 40 GB).
2. Inspect the actual file names and folder layout. The converter assumes `session<k>_participant<k>_gesture<k>_trial<k>`. If the real names differ, adjust the renaming in `scripts/convert_dat_to_mat.py` and add a test with the real pattern.
3. Run `convert_dat_to_mat.py --flatten --channels <28 indices>` and check `manifest.json` (sampling rate, units, channel names, record count) against the dataset description (28 of 32 channels, 2048 Hz, 15351 samples of 10240 × channels).
4. Run `prepare_folders.py --task all`.

**Acceptance:** per-task folder file counts match the thesis (for example 4515 samples for `5gest`, 9030 for `10gest`, 3570 for `10part`); the converter reports no NaNs, or the NaN count is explained.

### B2. Cross-session results (the table the README is missing)
Run for each task `10gest 5gest 17participants 10part 5part`, with `--seeds 1,2,3,4,5`:

| Method | Command |
|---|---|
| fixed SVM | `emg-gesture svm --task T --split session` |
| tuned SVM | same plus `--set svm.grid_search=true` |
| features | `emg-gesture features --task T --split session` (also `rf`) |
| ViT pretrained | `emg-gesture vit-google --task T --split session` |
| ViT scratch | `emg-gesture vit-scratch --task T --split session` |
| ViT spectrogram | each ViT command plus `--set model.input_repr=spectrogram --set augment.enabled=false` |

Participant-label tasks are forced to a random split; report them separately and say so.

**Acceptance:** every cell has an `aggregate.json`; `scripts/make_results_table.py` output is pasted into the README with mean ± std; the README states the split, seeds, software versions and hardware; the thesis tables stay labelled "reported in the thesis, not reproduced".

### B3. Questions the results must answer
- Does the ViT still beat the tuned SVM and the feature baselines? Report the answer even if it is no.
- Does `spectrogram` beat `flatten` with augmentation off?
- Is the gap larger than the seed std? Add a paired test (or confidence intervals) across seeds.
- Optional: a grouped (participant-disjoint) run on `10gest`/`5gest`, to measure generalisation to new users.

### B4. Pinned environment
1. Create a Python 3.10 environment; `pip install -r requirements.txt && pip install -e .`.
2. Run `REQUIRE_TORCH=1 pytest -q -rs`; fix or drop pins that fail.
3. If the thesis-era set cannot be made to work, remove it and keep a lock file generated from versions that pass CI.

**Acceptance:** the suite passes in that environment, or the README no longer offers a reproducible-pins path it has not tested.

### B5. Transformers 4.x
Run the `vit-google` and `vit-scratch` smoke tests, and the captioning repo's tests, on transformers 4.30 and 4.4x. Either widen the claim in the README or narrow the version cap in `pyproject.toml`.

## Part C: needs your accounts

| # | Task | Acceptance check |
|---|---|---|
| C1 | Review the diff, commit, and push `vit-thesis` (tag `pre-cleanup` already exists) | Both CI jobs are green on GitHub, including the required torch job. |
| C2 | Create a remote for `vit-gpt2-captioning`, make the first commit and push | Its CI job is green with `REQUIRE_TORCH=1`. |
| C3 | Make `test-torch` a required status check in branch protection | A deliberately failing test blocks a pull request. |

## Order
A1, A3, A4 first (small), then A2 and A5. Do C1 early so the CI result is known before the long runs. Then B1 to B2, B3 for the write-up, and B4/B5 last. A6 any time.

## Out of scope for this plan
ONNX export, latency measurement, FastAPI serving, a model card and a demo. They only make sense once B2 produces results worth shipping.
