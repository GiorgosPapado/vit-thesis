# vit-thesis

Code for my MSc thesis, *Vision Transformer for Gesture Recognition from EMG Signals* (MSc Digital Media & Computational Intelligence, Aristotle University of Thessaloniki): Vision Transformers (fine-tuned and trained from scratch) compared against SVM baselines across gesture-count and participant splits. The repository also holds a separate pipeline for ViT-GPT2 image captioning on a spatial COCO dataset.

> TODO: one sentence on the research question and the headline result.

## Repository layout

```
01_dataset/      captioning side work (not the EMG thesis): COCO/VOX dataset scripts
                 (copy_files.py, remove_empty.py, python_train_5000.py), scratch checks
                 (tester.py, testexport.py) and notebooks (vitgpt2.ipynb, test_gpt.ipynb)
02_training/     captioning side work: ViT-GPT2 training (train.py, training_version.ipynb)
03_experiments/  the EMG thesis experiments
  folders_creation.ipynb           dataset folder preparation
  code_reproduction_max_samples.ipynb
  svm_*.ipynb                      SVM baselines
  vit_google_*.ipynb               fine-tuning google ViT checkpoints
  vit_train_from_scratch_*.ipynb   ViT trained from scratch
  pca_components_search.ipynb      PCA analysis
  mlflow_plotting.ipynb            result plots from MLflow
requirements.txt         direct dependencies
requirements-freeze.txt  full original environment freeze
```

The `01_dataset/` scripts take `--data-dir` (or the `DATA_DIR` env var) for the dataset root. Outputs were stripped from the large notebooks to keep the repository small, so re-run them to regenerate results.

Notebook suffixes mean the experimental split: `5gest` / `10gest` (5 or 10 gesture classes), `5part` / `10part` / `17participants` (number of participants).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins direct dependencies without CUDA tags; the original freeze used the CUDA 11.8 builds (`+cu118`) of torch/torchvision. For GPU, install matching wheels from https://download.pytorch.org/whl/cu118 first. `scikit-learn`, `matplotlib` and `mlflow` are imported by the notebooks but were absent from the freeze, so they are unpinned.

## Reproduce

1. Build the dataset: `01_dataset/` scripts (see comments at the top of each).
2. Run the notebooks in `03_experiments/` in the order: `folders_creation` > `svm_*` > `vit_google_*` > `vit_train_from_scratch_*` > `mlflow_plotting`.

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
