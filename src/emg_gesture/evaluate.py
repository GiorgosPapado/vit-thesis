"""Test-set reporting: accuracy, chance levels, per-class accuracy, confusion matrix."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def report_eval(y_true, y_pred, labels, names=None, title: str = "Test set", plot: bool = True,
                show: bool = False):
    """Print and return a dict with accuracy, chance levels, per-class accuracy, support, cm.

    When ``plot`` is true the dict also has ``fig`` (confusion matrix figure).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)
    names = [str(n) for n in (labels if names is None else names)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    per_class = np.divide(cm.diagonal(), support, out=np.full(len(labels), np.nan), where=support > 0)
    acc = float((y_true == y_pred).mean())
    chance_uniform = 1.0 / len(labels)
    chance_majority = float(support.max() / support.sum())
    print(f"{title} accuracy: {acc:.4f}")
    print(f"Chance level: uniform 1/K = {chance_uniform:.4f} (K={len(labels)}), majority class = {chance_majority:.4f}")
    print("Per-class accuracy:")
    for n, a, s in zip(names, per_class, support):
        print(f"  class {n}: {a:.3f}  (n={s})")
    result = {
        "accuracy": acc,
        "chance_uniform": chance_uniform,
        "chance_majority": chance_majority,
        "per_class_accuracy": per_class,
        "support": support,
        "confusion_matrix": cm,
    }
    if plot:
        import matplotlib.pyplot as plt

        size = max(5, 0.5 * len(labels) + 3)
        fig, ax = plt.subplots(figsize=(size, size))
        ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, colorbar=False, xticks_rotation=90)
        ax.set_title(f"{title}: confusion matrix")
        fig.tight_layout()
        result["fig"] = fig
        if show:
            plt.show()
    return result
