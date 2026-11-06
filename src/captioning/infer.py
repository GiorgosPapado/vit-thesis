"""Caption images with a trained model and optionally export it (merges tester.py / testexport.py tail).

    python -m captioning.infer --model-dir OUT --images DIR_OR_FILE [--max-new-tokens 50] [--export-dir EXPORT]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from .paths import resolve_dir

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Caption images with a trained ViT-GPT2 model")
    p.add_argument("--model-dir", default=None, help="trained model dir (env: OUTPUT_DIR)")
    p.add_argument("--images", required=True, help="image file or directory")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--max-sentences", type=int, default=None, help="keep only the first N sentences")
    p.add_argument("--export-dir", default=None, help="also save model+processor+tokenizer here")
    return p


def collect_images(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"images path does not exist: {p}")


def first_sentences(text: str, n: Optional[int]) -> str:
    if n is None:
        return text
    return ".".join(text.split(".")[:n])


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    model_dir = resolve_dir(args.model_dir, "OUTPUT_DIR", must_exist=True)
    images = collect_images(args.images)
    from transformers import pipeline

    captioner = pipeline("image-to-text", model=str(model_dir), max_new_tokens=args.max_new_tokens)
    for img in images:
        out = captioner(str(img))
        text = out[0].get("generated_text", "") if out else ""
        print(f"{img.name}\t{first_sentences(text, args.max_sentences)}")
    if args.export_dir:
        Path(args.export_dir).mkdir(parents=True, exist_ok=True)
        captioner.model.save_pretrained(args.export_dir)
        captioner.image_processor.save_pretrained(args.export_dir)
        captioner.tokenizer.save_pretrained(args.export_dir)


if __name__ == "__main__":
    main()
