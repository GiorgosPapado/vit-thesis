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
from .features import features_pipeline
from .models import svm_pipeline
from .runs import load_config, resolve_split, unique_run_dir
from .seed import set_seed

log = logging.getLogger("emg_gesture")

TASKS = ("10gest", "5gest", "17participants", "10part", "5part")


def aggregate_runs(run_dirs, out_dir, task, method, seeds) -> Path:
    """Write ``aggregate.json`` (mean/std of test accuracy and macro-F1 over per-seed runs) to a new dir."""
    per_seed, first = [], None
    for seed, rd in zip(seeds, run_dirs):
        m = json.loads((Path(rd) / "metrics.json").read_text(encoding="utf8"))
        first = first or m
        per_seed.append({"seed": seed, "run_dir": Path(rd).name, "test_accuracy": m["test_accuracy"],
                         "macro_f1": m["macro_f1"]})
    agg = {"task": task, "method": method, "seeds": list(seeds), "n_runs": len(per_seed), "per_seed": per_seed,
           "std_ddof": 1 if len(per_seed) > 1 else 0}
    if first is not None:  # identical across seeds: same config, only the seed differs
        for key in ("split_mode", "generalises_to_new_subjects", "train_sessions", "test_sessions"):
            agg[key] = first.get(key)
        if agg["split_mode"] == "session":
            agg["std_note"] = ("session mode: train/test files are fixed, so the std over seeds reflects "
                               "model initialisation and the validation split only, not data sampling")
    for key in ("test_accuracy", "macro_f1"):
        v = np.array([r[key] for r in per_seed], dtype=float)
        agg[key] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    path = unique_run_dir(out_dir, task, f"{method}-aggregate", "seeds")
    with open(path / "aggregate.json", "w", encoding="utf8") as f:
        json.dump(agg, f, indent=2)
    log.info("aggregate over seeds %s: accuracy %.4f +- %.4f", list(seeds), agg["test_accuracy"]["mean"],
             agg["test_accuracy"]["std"])
    return path


def _listify_grid(grid_cfg: dict) -> dict:
    """Wrap scalar grid values (e.g. ``--set svm.grid.C=1``) into one-element lists."""
    return {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in grid_cfg.items()}


def _svm_param_grid(grid_cfg: dict) -> list:
    """Config grid -> list of param dicts; the linear kernel gets no ``gamma`` (it is ignored there)."""
    grid_cfg = _listify_grid(grid_cfg)
    base = {k: v for k, v in grid_cfg.items() if k not in ("kernel", "gamma")}
    out = []
    for kern in grid_cfg.get("kernel", ["linear"]):
        d = {**base, "kernel": [kern]}
        if kern != "linear" and "gamma" in grid_cfg:
            d["gamma"] = grid_cfg["gamma"]
        out.append({f"svc__{k}": v for k, v in d.items()})
    return out


def _svm_fit_eval(cfg, pipe, X_train, y_train, X_test, cv, cv_groups):
    """Fit the SVM pipeline (optionally grid search); return predictions, CV scores, metrics fields."""
    from sklearn.model_selection import GridSearchCV, cross_val_score

    grid_search = bool(cfg["svm"].get("grid_search", False))
    hyper = "fixed (no grid search)"
    extra = {}
    if grid_search:
        grid = _svm_param_grid(cfg["svm"]["grid"])  # pca_components stays fixed
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", refit=True)
        gs.fit(X_train, y_train, groups=cv_groups)  # train files only; test set untouched until predict
        cv_scores = None  # selection-biased per-fold scores are not emitted as cv_scores/cv_mean
        hyper = "grid_search"
        extra = {"best_params": {k.removeprefix("svc__"): v for k, v in gs.best_params_.items()},
                 "grid": _listify_grid(cfg["svm"]["grid"]),
                 "best_cv_score": float(gs.best_score_), "cv_is_selection_biased": True}
        y_pred = gs.best_estimator_.predict(X_test)
    else:
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, groups=cv_groups, scoring="accuracy")
        pipe.fit(X_train, y_train)  # test set untouched until here
        y_pred = pipe.predict(X_test)
    # grid runs: kernel/C/gamma are tuned, so record only the fixed PCA size and the grid (best_params: rest)
    svm_config = dict(cfg["svm"])
    if grid_search:
        svm_config = {"pca_components": svm_config["pca_components"], "grid": extra["grid"]}
    fields = {
        "svm_uses_full_window": True,  # unlike the ViT input, which is truncated to 3x224x224 values
        "svm_hyperparameters": hyper,
        **extra,
        "svm_config": svm_config,
    }
    return y_pred, cv_scores, fields


def _features_fit_eval(cfg, pipe, X_train, y_train, X_test, cv, cv_groups):
    from sklearn.model_selection import cross_val_score

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, groups=cv_groups, scoring="accuracy")
    pipe.fit(X_train, y_train)  # test set untouched until here
    y_pred = pipe.predict(X_test)
    fcfg = cfg["features"]
    clf = fcfg.get("classifier", "lda")
    fields = {
        "feature_names": list(fcfg["features"]),
        "classifier": clf,
        "classifier_params": dict(fcfg[clf]) if isinstance(fcfg.get(clf), dict) else {},
    }
    return y_pred, cv_scores, fields


def run_svm(cfg: dict, out_dir="runs") -> Path:
    """Run the SVM baseline for a resolved config and write the run directory."""
    return _run_sklearn(cfg, out_dir, "svm", svm_pipeline, _svm_fit_eval, "SVM")


def run_features(cfg: dict, out_dir="runs") -> Path:
    """Run the handcrafted-feature baseline for a resolved config and write the run directory."""
    return _run_sklearn(cfg, out_dir, "features", features_pipeline, _features_fit_eval, "features")


def _run_sklearn(cfg, out_dir, method, build_pipeline, fit_eval, title_name) -> Path:
    """Shared sklearn run: load, split, CV, test eval and artifacts (config, metrics, confusion, per-class)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GroupKFold, StratifiedKFold

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
        test_sessions=cfg.get("test_sessions", (3,)), val_sessions=None)
    X_train = np.array([load_flattened(files[i], cfg["target_length"]) for i in train_idx])
    X_test = np.array([load_flattened(files[i], cfg["target_length"]) for i in test_idx])
    y_train, y_test = labels[train_idx], labels[test_idx]
    n_channels = X_train.shape[1] // cfg["target_length"]

    pipe = build_pipeline(cfg, n_channels)
    if cfg["split"] == "grouped":
        cv_groups = np.array([extract_group_from_filename(files[i]) for i in train_idx], dtype=object)
        cv = GroupKFold(n_splits=cfg["cv_splits"])
    else:
        cv_groups = None
        cv = StratifiedKFold(n_splits=cfg["cv_splits"], shuffle=True, random_state=seed)
    y_pred, cv_scores, fields = fit_eval(cfg, pipe, X_train, y_train, X_test, cv, cv_groups)
    classes = np.unique(np.concatenate([y_train, y_test]))
    res = report_eval(y_test, y_pred, labels=classes, title=f"Test set ({title_name}, {task})")

    run_dir = unique_run_dir(out_dir, task, method, seed)
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
        **fields,
        "test_accuracy": res["accuracy"],
        "macro_f1": res["macro_f1"],
        "chance_uniform": res["chance_uniform"],
        "chance_majority": res["chance_majority"],
    }
    if cv_scores is not None:
        metrics["cv_scores"] = [float(s) for s in cv_scores]
        metrics["cv_mean"] = float(np.mean(cv_scores))
        metrics["cv_is_selection_biased"] = False
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
                           ("features", "handcrafted time-domain features + LDA/RF/SVM baseline"),
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
    elif args.command == "features":
        def runner(c):
            return run_features(c, args.out_dir)
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
