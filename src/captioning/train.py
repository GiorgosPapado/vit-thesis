"""Train ViT-GPT2 captioning (merges 01_dataset/python_train_5000.py and 02_training/train.py).

    python -m captioning.train --data-dir DIR --output-dir OUT [--config configs/captioning.yaml]
        [--resume-from-checkpoint OUT/checkpoint-8000] [--set num_train_epochs=10]

Expected layout under DATA_DIR: image folders (jpg) and description folders holding
``<image stem>_desc.txt``; folder names are set in the YAML (train_images, val_images, train_desc, val_desc).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import load_config
from .data import load_captions
from .paths import resolve_dir

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ViT-GPT2 image captioning")
    p.add_argument("--config", default=None, help="YAML config (default configs/captioning.yaml if present)")
    p.add_argument("--data-dir", default=None, help="dataset root (env: DATA_DIR)")
    p.add_argument("--output-dir", default=None, help="output root (env: OUTPUT_DIR)")
    p.add_argument("--seed", type=int, default=None, help="overrides config seed")
    p.add_argument("--resume-from-checkpoint", default=None, help="checkpoint dir, passed to trainer.train()")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return p


def eval_strategy_kwargs(arg_fields: Sequence[str], strategy: str = "epoch") -> Dict[str, str]:
    """transformers>=4.41 renamed evaluation_strategy to eval_strategy; pick whichever exists."""
    if "eval_strategy" in arg_fields:
        return {"eval_strategy": strategy}
    return {"evaluation_strategy": strategy}


def build_training_kwargs(cfg: Dict[str, Any], output_dir: Path, arg_fields: Sequence[str]) -> Dict[str, Any]:
    """Seq2SeqTrainingArguments kwargs (no resume_from_checkpoint: that belongs to trainer.train)."""
    kwargs: Dict[str, Any] = dict(
        predict_with_generate=True,
        per_device_train_batch_size=int(cfg.get("batch_size", 24)),
        per_device_eval_batch_size=int(cfg.get("batch_size", 24)),
        num_train_epochs=float(cfg.get("num_train_epochs", 10)),
        output_dir=str(output_dir),
        seed=int(cfg.get("seed", 42)),
    )
    kwargs.update(eval_strategy_kwargs(arg_fields, cfg.get("eval_strategy", "epoch")))
    return kwargs


def set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def build_records(image_dir: Path, desc_dir: Path, word_limit: int = 50) -> Dict[str, list]:
    """Pair images with captions by stem; images without a caption file are skipped."""
    captions = load_captions(desc_dir, word_limit)
    paths, texts = [], []
    for img in list_images(image_dir):
        cap = captions.get(img.stem)
        if cap is not None:
            paths.append(str(img))
            texts.append(cap)
    return {"image_path": paths, "caption": texts}


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config, args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("seed", 42)
    cfg["data_dir"] = str(resolve_dir(args.data_dir or cfg.get("data_dir"), "DATA_DIR"))
    cfg["output_dir"] = str(resolve_dir(args.output_dir or cfg.get("output_dir"), "OUTPUT_DIR"))
    cfg["resume_from_checkpoint"] = args.resume_from_checkpoint or cfg.get("resume_from_checkpoint")
    return cfg


def build_trainer(trainer_cls, processor, **kwargs):
    """Instantiate ``trainer_cls`` passing the processor as ``processing_class`` (new transformers) or
    ``tokenizer`` (older versions), chosen by inspecting the constructor signature."""
    import inspect

    params = inspect.signature(trainer_cls.__init__).parameters
    key = "processing_class" if "processing_class" in params else "tokenizer"
    return trainer_cls(**{key: processor}, **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = resolve_settings(build_parser().parse_args(argv))
    os.environ.setdefault("WANDB_DISABLED", "true")
    import evaluate
    import nltk
    import numpy as np
    from datasets import Dataset
    from PIL import Image
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
        default_data_collator,
    )

    set_seeds(cfg["seed"])
    data_dir, out_dir = Path(cfg["data_dir"]), Path(cfg["output_dir"])
    enc_name = cfg.get("image_encoder_model", "google/vit-base-patch16-224-in21k")
    dec_name = cfg.get("text_decoder_model", "gpt2")
    max_len = int(cfg.get("max_target_length", 200))
    word_limit = int(cfg.get("word_limit", 50))

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(enc_name, dec_name)
    feature_extractor = AutoImageProcessor.from_pretrained(enc_name)
    tokenizer = AutoTokenizer.from_pretrained(dec_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    def preprocess(batch):
        images = []
        for f in batch["image_path"]:
            with Image.open(f) as im:
                images.append(im.convert("RGB"))
        labels = tokenizer(batch["caption"], padding="max_length", max_length=max_len).input_ids
        return {"labels": labels, "pixel_values": feature_extractor(images=images, return_tensors="np").pixel_values}

    def make(images_key, desc_key):
        rec = build_records(data_dir / cfg[images_key], data_dir / cfg[desc_key], word_limit)
        return Dataset.from_dict(rec).map(preprocess, batched=True)

    train_ds = make("train_images", "train_desc")
    val_ds = make("val_images", "val_desc")

    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)
    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        dp = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
        dp = ["\n".join(nltk.sent_tokenize(x.strip())) for x in dp]
        dl = ["\n".join(nltk.sent_tokenize(x.strip())) for x in dl]
        res = metric.compute(predictions=dp, references=dl, use_stemmer=True)
        res = {k: round(v * 100, 4) for k, v in res.items()}
        res["gen_len"] = float(np.mean([np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]))
        return res

    fields = [f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)]
    args = Seq2SeqTrainingArguments(**build_training_kwargs(cfg, out_dir, fields))
    trainer = build_trainer(
        Seq2SeqTrainer,
        feature_extractor,
        model=model,
        args=args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )
    trainer.train(resume_from_checkpoint=cfg["resume_from_checkpoint"])
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    feature_extractor.save_pretrained(str(out_dir))


if __name__ == "__main__":
    main()
