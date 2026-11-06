"""Command line entry point: ``emg-gesture svm --task 10gest --split grouped --seed 42``."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import yaml

from .data import (
    extract_group_from_filename,
    extract_label_from_filename,
    list_mat_files,
    load_flattened,
    split_files_info,
    split_report,
)
from .evaluate import report_eval
from .models import svm_pipeline
from .runs import load_config, resolve_split, unique_run_dir
from .seed import set_seed

log = logging.getLogger("emg_gesture")

TASKS = ("10gest", "5gest", "17participants", "10part", "5part")


def aggregate_runs(run_dirs, out_dir, task, method, seeds) -> Path:
    """Write ``aggregate.json`` (mean/std of test accuracy and macro-F1 over per-seed runs) to a new dir."""
    per_seed = []
    for seed, rd in zip(seeds, run_dirs):
        m = json.loads((Path(rd) / "metrics.json").read_text(encoding="utf8"))
        per_seed.append({"seed": seed, "run_dir": Path(rd).name, "test_accuracy": m["test_accuracy"],
                         "macro_f1": m["macro_f1"]})
    agg = {"task": task, "method": method, "seeds": list(seeds), "n_runs": len(per_seed), "per_seed": per_seed,
           "std_ddof": 1 if len(per_seed) > 1 else 0}
    for key in ("test_accuracy", "macro_f1"):
        v = np.array([r[key] for r in per_seed], dtype=float)
        agg[key] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    path = unique_run_dir(out_dir, task, f"{method}-aggregate", "seeds")
    with open(path / "aggregate.json", "w", encoding="utf8") as f:
        json.dump(agg, f, indent=2)
    log.info("aggregate over seeds %s: accuracy %.4f +- %.4f", list(seeds), agg["test_accuracy"]["mean"],
             agg["test_accuracy"]["std"])
    return path


def run_svm(cfg: dict, out_dir="runs") -> Path:
    """Run the SVM baseline for a resolved config and write the run directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

    cfg = copy.deepcopy(cfg)
    seed = int(cfg["seed"])
    set_seed(seed, bool(cfg.get("deterministic", False)))
    task, label_kind = cfg["task"], cfg["label"]
    forced_reason = resolve_split(cfg, task)

    files = list_mat_files(cfg["data_dir"])
    if not files:
        raise FileNotFoundError(f"no .mat files under {cfg['data_dir']!r}")
    parse = extract_label_from_filename if label_kind == "gesture" else extract_group_from_filename
    labels = [parse(f) for f in files]
    if any(lab is None for lab in labels):
        raise ValueError(f"could not parse a {label_kind} label from every filename")
    labels = np.array(labels)
    n_found = len(np.unique(labels))
    if cfg.get("n_classes") not in (None, n_found):
        log.warning("config n_classes=%s but %d distinct labels found in %s", cfg["n_classes"], n_found,
                    cfg["data_dir"])

    train_idx, val_idx, test_idx, split_info = split_files_info(
        files, labels, mode=cfg["split"], test_size=cfg["test_size"], val_size=0.0, seed=seed,
        stratify=cfg.get("stratify", True), train_sessions=cfg.get("train_sessions", (1, 2)),
        test_sessions=cfg.get("test_sessions", (3,)))
    X_train = np.array([load_flattened(files[i], cfg["target_length"]) for i in train_idx])
    X_test = np.array([load_flattened(files[i], cfg["target_length"]) for i in test_idx])
    y_train, y_test = labels[train_idx], labels[test_idx]
    n_channels = X_train.shape[1] // cfg["target_length"]

    pipe = svm_pipeline(cfg, n_channels)
    if cfg["split"] == "grouped":
        cv_groups = np.array([extract_group_from_filename(files[i]) for i in train_idx], dtype=object)
        cv = GroupKFold(n_splits=cfg["cv_splits"])
    else:
        cv_groups = None
        cv = StratifiedKFold(n_splits=cfg["cv_splits"], shuffle=True, random_state=seed)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, groups=cv_groups, scoring="accuracy")
    pipe.fit(X_train, y_train)  # test set untouched until here
    y_pred = pipe.predict(X_test)
    classes = np.unique(np.concatenate([y_train, y_test]))
    res = report_eval(y_test, y_pred, labels=classes, title=f"Test set (SVM, {task})")

    run_dir = unique_run_dir(out_dir, task, "svm", seed)
    with open(run_dir / "config.resolved.yaml", "w", encoding="utf8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    metrics = {
        "task": task,
        "split": cfg["split"],
        "split_forced_reason": forced_reason,
        **split_report(labels, train_idx, val_idx, test_idx, split_info),
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes_seen": int(len(classes)),
        "test_accuracy": res["accuracy"],
        "macro_f1": res["macro_f1"],
        "chance_uniform": res["chance_uniform"],
        "chance_majority": res["chance_majority"],
        "cv_scores": [float(s) for s in cv_scores],
        "cv_mean": float(np.mean(cv_scores)),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)
    res["fig"].savefig(run_dir / "confusion.png", dpi=120)
    plt.close(res["fig"])
    with open(run_dir / "per_class.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["label", "support", "accuracy"])
        for lab, sup, acc in zip(classes, res["support"], res["per_class_accuracy"]):
            w.writerow([int(lab), int(sup), "" if np.isnan(acc) else float(acc)])
    log.info("test accuracy %.4f, wrote %s", res["accuracy"], run_dir)
    return run_dir


def _add_common_options(p) -> None:
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--split", choices=("grouped", "random", "session"), default=None, help="default: from config")
    p.add_argument("--seed", type=int, default=None, help="default: from config")
    p.add_argument("--seeds", default=None, metavar="S1,S2,...",
                   help="comma list; run each seed and write aggregate.json (mean/std of accuracy, macro-F1)")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value (dotted keys ok, e.g. svm.kernel=rbf); repeatable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emg-gesture", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, helptext in (("svm", "train/evaluate the SVM baseline"),
                           ("vit-google", "fine-tune the pretrained Google ViT"),
                           ("vit-scratch", "train a small ViT from scratch")):
        _add_common_options(sub.add_parser(name, help=helptext))
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config_dir, args.command.replace("-", "_"), args.task, args.overrides)
    if args.seed is not None and args.seeds is not None:
        raise SystemExit("use either --seed or --seeds, not both")
    seeds = None
    if args.seeds is not None:
        try:
            seeds = [int(t) for t in args.seeds.split(",") if t.strip()]
        except ValueError:
            raise SystemExit("--seeds must be a comma list of integers") from None
        if not seeds:
            raise SystemExit("--seeds is empty")
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.split is not None:
        cfg["split"] = args.split
    if cfg.get("data_dir") is None:
        raise SystemExit("data_dir is not set (task config or --set data_dir=...)")
    cfg["data_dir"] = os.fspath(cfg["data_dir"])
    if args.command == "svm":
        def runner(c):
            return run_svm(c, args.out_dir)
    else:
        from .train_vit import run_vit

        def runner(c):
            return run_vit(c, args.command.split("-", 1)[1], args.out_dir)
    if seeds is None:
        runner(cfg)
    else:
        dirs = [runner({**cfg, "seed": s}) for s in seeds]
        aggregate_runs(dirs, args.out_dir, cfg["task"], args.command, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
