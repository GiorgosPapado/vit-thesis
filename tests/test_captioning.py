"""Torch-free tests for the captioning side package."""
import dataclasses

import pytest

from captioning import cli, data, infer, train
from captioning.config import load_config
from captioning.paths import resolve_dir


def test_resolve_dir_arg_env_and_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    with pytest.raises(ValueError):
        resolve_dir(None, "DATA_DIR")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert resolve_dir(None, "DATA_DIR") == tmp_path
    assert resolve_dir("x", "DATA_DIR").name == "x"
    with pytest.raises(FileNotFoundError):
        resolve_dir(str(tmp_path / "nope"), "DATA_DIR", must_exist=True)


def test_copy_subset(tmp_path):
    src, dst = tmp_path / "s", tmp_path / "d" / "x"
    src.mkdir()
    for i in range(5):
        (src / f"f{i}.txt").write_text(str(i))
    copied = data.copy_subset(src, dst, 3)
    assert [p.name for p in copied] == ["f0.txt", "f1.txt", "f2.txt"]
    assert len(list(dst.iterdir())) == 3
    with pytest.raises(FileNotFoundError):
        data.copy_subset(tmp_path / "none", dst)


def test_remove_empty(tmp_path):
    (tmp_path / "a").write_text("")
    (tmp_path / "b").write_text("x")
    assert [p.name for p in data.remove_empty_files(tmp_path)] == ["a"]
    assert (tmp_path / "a").exists()  # dry run
    data.remove_empty_files(tmp_path, dry_run=False)
    assert not (tmp_path / "a").exists() and (tmp_path / "b").exists()


def test_load_captions_and_records(tmp_path):
    desc, img = tmp_path / "desc", tmp_path / "img"
    desc.mkdir()
    img.mkdir()
    (desc / "p1_desc.txt").write_text("1 A cat.\n2 A dog.\n")
    (desc / "p2_desc.txt").write_text("x")
    (img / "p1.jpg").write_bytes(b"")
    (img / "p3.jpg").write_bytes(b"")
    caps = data.load_captions(desc)
    assert caps["p1"] == "A cat.. A dog.."
    rec = train.build_records(img, desc)
    assert len(rec["image_path"]) == 1 and rec["image_path"][0].endswith("p1.jpg")
    assert data.limit_words("a b c", 2) == "a b..."


def test_eval_strategy_both_names(tmp_path):
    assert train.eval_strategy_kwargs(["evaluation_strategy"]) == {"evaluation_strategy": "epoch"}
    assert train.eval_strategy_kwargs(["eval_strategy", "x"]) == {"eval_strategy": "epoch"}
    kw = train.build_training_kwargs({"seed": 7}, tmp_path, ["evaluation_strategy"])
    assert kw["seed"] == 7 and "resume_from_checkpoint" not in kw


def test_load_config_overrides(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("seed: 1\nbatch_size: 2\n")
    assert load_config(str(f), ["batch_size=8"]) == {"seed": 1, "batch_size": 8}
    with pytest.raises(ValueError):
        load_config(str(f), ["bad"])


def test_train_settings_args(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    ns = train.build_parser().parse_args(["--output-dir", str(tmp_path / "o"), "--seed", "3", "--set", "batch_size=4"])
    cfg = train.resolve_settings(ns)
    assert cfg["seed"] == 3 and cfg["batch_size"] == 4
    assert cfg["data_dir"] == str(tmp_path / "d") and cfg["resume_from_checkpoint"] is None


def test_cli_subset_and_remove_empty(tmp_path, capsys):
    src = tmp_path / "src" / "val2014_desc"
    src.mkdir(parents=True)
    (src / "a").write_text("")
    (src / "b").write_text("1")
    cli.main(["subset", "--data-dir", str(tmp_path / "src"), "--dest-dir", str(tmp_path / "dst"), "-n", "1"])
    assert (tmp_path / "dst" / "val2014_desc" / "a").exists()
    cli.main(["remove-empty", "--data-dir", str(tmp_path / "src"), "--subdir", "val2014_desc", "--delete"])
    assert not (src / "a").exists()
    assert "deleted 1" in capsys.readouterr().out


def test_infer_helpers(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.txt").write_bytes(b"")
    assert [p.name for p in infer.collect_images(str(tmp_path))] == ["a.jpg"]
    assert infer.first_sentences("A. B. C.", 2) == "A. B"
    with pytest.raises(FileNotFoundError):
        infer.collect_images(str(tmp_path / "zz"))


def test_transformers_training_args_accept_kwargs(tmp_path):
    pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    fields = [f.name for f in dataclasses.fields(tf.Seq2SeqTrainingArguments)]
    args = tf.Seq2SeqTrainingArguments(**train.build_training_kwargs({"seed": 5}, tmp_path, fields))
    assert args.seed == 5


def test_build_trainer_picks_processing_class_or_tokenizer():
    class New:
        def __init__(self, model=None, processing_class=None):
            self.kw = {"model": model, "processing_class": processing_class}

    class Old:
        def __init__(self, model=None, tokenizer=None):
            self.kw = {"model": model, "tokenizer": tokenizer}

    assert train.build_trainer(New, "proc", model="m").kw == {"model": "m", "processing_class": "proc"}
    assert train.build_trainer(Old, "proc", model="m").kw == {"model": "m", "tokenizer": "proc"}
