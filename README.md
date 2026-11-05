# vit-thesis

Code for my MSc thesis: classifying VR controller gestures with Vision Transformers (ViT), compared against an SVM baseline, plus the image-captioning pipeline (ViT-GPT2 on a spatial COCO dataset) that grew out of the same work.

> TODO: one sentence on the research question and the headline result.

## Repository layout

```
01_dataset/      dataset creation scripts (copy_files.py, remove_empty.py, python_train_5000.py; tester.py, testexport.py, vitgpt2.ipynb, test_gpt.ipynb are checks and the ViT-GPT2 notebook)
02_training/     ViT-GPT2 training on the spatial COCO dataset (train.py, training_version.ipynb)
03_experiments/
  svm_*.ipynb                      SVM baselines
  vit_google_*.ipynb               fine-tuning google ViT checkpoints
  vit_train_from_scratch_*.ipynb   ViT trained from scratch
  pca_components_search.ipynb      PCA analysis
  mlflow_plotting.ipynb            result plots from MLflow
```

Notebook suffixes mean the experimental split: `5gest` / `10gest` (5 or 10 gesture classes), `5part` / `10part` / `17participants` (number of participants).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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
