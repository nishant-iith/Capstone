"""
Generate best-100 CLAHE/content-quality showcase panels.

Each output panel shows:
  Unstained input | Virtual H&E (balanced final ensemble) | Real H&E target

The source pairs are the first N rows of the final content-quality positive
CLAHE top-1000 CSV. That CSV is already sorted by the final registration/content
quality policy used for v22A/v21B training.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from best_stain_app import (
    ModelPaths,
    default_model_paths,
    load_bundle,
    pil_to_tensor,
    run_pipeline,
    tensor_to_pil,
)


SCORES_CSV = Path("data/processed/content_quality_csvs/content_quality_minrgb0.50_positive_top1000_scores.csv")
OUT_DIR = Path("showcase_images/best100_clahe_balanced")
FEATURED_RANKS = [1, 2, 3, 4, 5, 6]


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def load_rows(limit: int) -> list[dict[str, str]]:
    with SCORES_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if row.get("status", "ok") == "ok"]
    rows.sort(key=lambda r: float(r["content_quality_score"]), reverse=True)
    return rows[:limit]


def image_to_tensor(image: Image.Image, size: int) -> tuple[torch.Tensor, Image.Image]:
    return pil_to_tensor(image.convert("RGB"), size)


def label_bar(width: int, text: str, height: int = 56) -> Image.Image:
    bar = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small = font
    main, _, sub = text.partition("\n")
    draw.text((14, 8), main, fill=(20, 31, 46), font=font)
    if sub:
        draw.text((14, 33), sub, fill=(82, 95, 111), font=small)
    return bar


def resize_square(image: Image.Image, tile: int) -> Image.Image:
    return image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)


def make_panel(
    rank: int,
    row: dict[str, str],
    unstained: Image.Image,
    virtual: Image.Image,
    stained: Image.Image,
    tile: int,
) -> Image.Image:
    gap = 16
    margin = 22
    label_h = 56
    header_h = 72
    footer_h = 42
    width = margin * 2 + tile * 3 + gap * 2
    height = margin * 2 + header_h + label_h + tile + footer_h

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 23)
        meta_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        title_font = ImageFont.load_default()
        meta_font = title_font

    title = f"#{rank:03d}  {row['prefix']}"
    meta = (
        f"content-quality={float(row['content_quality_score']):.4f}  "
        f"CLAHE SSIM={float(row['ssim_full_rgb']):.4f}  "
        f"content={float(row['content_fraction']):.3f}"
    )
    draw.text((margin, margin), title, fill=(14, 23, 36), font=title_font)
    draw.text((margin, margin + 34), meta, fill=(82, 95, 111), font=meta_font)

    labels = [
        "Unstained Input\nregistered CLAHE pair",
        "Virtual H&E\nbalanced final ensemble",
        "Real H&E Target\nground truth",
    ]
    imgs = [unstained, virtual, stained]
    y_label = margin + header_h
    y_img = y_label + label_h
    for idx, (label, img) in enumerate(zip(labels, imgs)):
        x = margin + idx * (tile + gap)
        canvas.paste(label_bar(tile, label, label_h), (x, y_label))
        canvas.paste(resize_square(img, tile), (x, y_img))

    footer = "Generated from final CLAHE content-quality top-100 using balanced v20/v22A/v21B TTA4 model."
    draw.text((margin, height - margin - 18), footer, fill=(102, 112, 128), font=meta_font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--size", type=int, default=1024, choices=[512, 768, 1024])
    parser.add_argument("--tile", type=int, default=384)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--mode", default="balanced", choices=["balanced", "best_ssim", "v22_tta", "v20_tta"])
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    featured_dir = OUT_DIR / "featured"
    featured_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.limit)
    defaults = default_model_paths()
    paths = ModelPaths(defaults["v20"], defaults["v22"], defaults["v21b"])
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print(f"Loading final model bundle on {device} in {args.mode} mode...")
    bundle = load_bundle(paths, device, args.mode, progress=print)

    metadata_path = OUT_DIR / "metadata.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "prefix",
                "panel",
                "featured_panel",
                "content_quality_score",
                "ssim_full_rgb",
                "ssim_content_gray",
                "content_fraction",
                "delta_vs_old",
            ],
            lineterminator="\n",
        )
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            panel_name = f"{idx:03d}_{slugify(row['prefix'])}.jpg"
            panel_path = OUT_DIR / panel_name
            featured_path = featured_dir / panel_name if idx in FEATURED_RANKS else None

            if panel_path.exists() and (featured_path is None or featured_path.exists()):
                print(f"[{idx:03d}/{len(rows)}] skip existing {panel_name}")
            else:
                print(f"[{idx:03d}/{len(rows)}] generating {row['prefix']}")
                unstained = Image.open(row["unstained"]).convert("RGB")
                stained = Image.open(row["stained"]).convert("RGB")
                tensor, resized_input = image_to_tensor(unstained, args.size)
                pred = run_pipeline(
                    bundle,
                    tensor,
                    device,
                    args.mode,
                    use_tta=not args.no_tta,
                    progress=lambda msg: print(f"  {msg}"),
                )
                virtual = tensor_to_pil(pred)
                stained_resized = stained.resize((args.size, args.size), Image.Resampling.BILINEAR)
                panel = make_panel(idx, row, resized_input, virtual, stained_resized, args.tile)
                panel.save(panel_path, quality=args.quality, optimize=True)
                if featured_path is not None:
                    panel.save(featured_path, quality=args.quality, optimize=True)

            writer.writerow(
                {
                    "rank": idx,
                    "prefix": row["prefix"],
                    "panel": str(panel_path),
                    "featured_panel": str(featured_path) if featured_path else "",
                    "content_quality_score": row["content_quality_score"],
                    "ssim_full_rgb": row["ssim_full_rgb"],
                    "ssim_content_gray": row["ssim_content_gray"],
                    "content_fraction": row["content_fraction"],
                    "delta_vs_old": row["delta_vs_old"],
                }
            )

    print(f"Wrote {metadata_path}")
    print(f"Wrote panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
