"""
Post-registration tissue-aware scoring.

Reads the CLAHE registration live/final CSV, computes tissue-only SSIM scores
for completed registered pairs, and writes ranking CSVs without modifying any
registered images.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from skimage import io
from skimage.metrics import structural_similarity as ssim_fn


DEFAULT_INPUT = "logs/registration_clahe_per_pair.csv"
DEFAULT_OUT = "data/processed/registered_clahe_tissue_scores.csv"
DEFAULT_OUT_DIR = "data/processed/tissue_ranked_csvs"


def load_rgb(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def tissue_mask(stained: np.ndarray, unstained: np.ndarray) -> np.ndarray:
    """Return a conservative tissue foreground mask from fixed and moving images."""
    masks = []
    for img in (stained, unstained):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0
        mean_rgb = img.astype(np.float32).mean(axis=2) / 255.0

        # Tissue is generally less white and/or more saturated than background.
        mask = ((saturation > 0.045) | (value < 0.93) | (mean_rgb < 0.92)).astype(np.uint8)
        masks.append(mask)

    mask = np.maximum(masks[0], masks[1])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Avoid scoring tiny uncertain fragments as a full tissue region.
    if mask.mean() < 0.01:
        return np.zeros_like(mask, dtype=bool)
    return mask.astype(bool)


def content_mask(stained: np.ndarray, unstained: np.ndarray, percentile: float = 75.0) -> np.ndarray:
    """High-information mask from edges/texture, useful when every patch is tissue."""
    s_gray = cv2.cvtColor(stained, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    u_gray = cv2.cvtColor(unstained, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    s_edge = cv2.Sobel(s_gray, cv2.CV_32F, 1, 0, ksize=3) ** 2
    s_edge += cv2.Sobel(s_gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    u_edge = cv2.Sobel(u_gray, cv2.CV_32F, 1, 0, ksize=3) ** 2
    u_edge += cv2.Sobel(u_gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    edge = np.sqrt(np.maximum(s_edge, u_edge))

    threshold = float(np.percentile(edge, percentile))
    if threshold <= 1e-6:
        return np.ones_like(edge, dtype=bool)

    mask = edge >= threshold
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def masked_gray_ssim(stained: np.ndarray, unstained: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    s_gray = cv2.cvtColor(stained, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    u_gray = cv2.cvtColor(unstained, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    full_score, score_map = ssim_fn(s_gray, u_gray, data_range=1.0, full=True)
    if not mask.any():
        return float(full_score), float(full_score)
    return float(full_score), float(score_map[mask].mean())


def masked_rgb_ssim(stained: np.ndarray, unstained: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    full_score = float(ssim_fn(stained, unstained, channel_axis=2, data_range=255))
    if not mask.any():
        return full_score, full_score

    channel_scores = []
    for channel in range(3):
        _, score_map = ssim_fn(
            stained[:, :, channel],
            unstained[:, :, channel],
            data_range=255,
            full=True,
        )
        channel_scores.append(float(score_map[mask].mean()))
    return full_score, float(np.mean(channel_scores))


def score_row(row: dict[str, object]) -> dict[str, object]:
    stained_path = str(row["stained"])
    unstained_path = str(row["unstained"])
    prefix = str(row["prefix"])

    try:
        stained = load_rgb(stained_path)
        unstained = load_rgb(unstained_path)
        mask = tissue_mask(stained, unstained)
        high_content = content_mask(stained, unstained)
        full_gray, tissue_gray = masked_gray_ssim(stained, unstained, mask)
        full_rgb, tissue_rgb = masked_rgb_ssim(stained, unstained, mask)
        _, content_gray = masked_gray_ssim(stained, unstained, high_content)
        _, content_rgb = masked_rgb_ssim(stained, unstained, high_content)
        tissue_fraction = float(mask.mean())
        content_fraction = float(high_content.mean())

        return {
            "prefix": prefix,
            "stained": stained_path,
            "unstained": unstained_path,
            "ssim_full_rgb": full_rgb,
            "ssim_full_gray": full_gray,
            "ssim_tissue_rgb": tissue_rgb,
            "ssim_tissue_gray": tissue_gray,
            "ssim_content_rgb": content_rgb,
            "ssim_content_gray": content_gray,
            "tissue_fraction": tissue_fraction,
            "content_fraction": content_fraction,
            "old_ssim": row.get("old_ssim", ""),
            "delta_vs_old": row.get("delta_vs_old", ""),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "prefix": prefix,
            "stained": stained_path,
            "unstained": unstained_path,
            "ssim_full_rgb": 0.0,
            "ssim_full_gray": 0.0,
            "ssim_tissue_rgb": 0.0,
            "ssim_tissue_gray": 0.0,
            "ssim_content_rgb": 0.0,
            "ssim_content_gray": 0.0,
            "tissue_fraction": 0.0,
            "content_fraction": 0.0,
            "old_ssim": row.get("old_ssim", ""),
            "delta_vs_old": row.get("delta_vs_old", ""),
            "status": f"error: {exc}",
        }


def read_input(path: str, limit: int | None) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[~df["status"].astype(str).str.startswith("error")].copy()
    if "stained" not in df.columns or "unstained" not in df.columns:
        raise ValueError("Input CSV must contain stained and unstained columns.")

    df = df.drop_duplicates("prefix", keep="last")
    df = df[df["stained"].map(os.path.exists) & df["unstained"].map(os.path.exists)]
    if limit is not None:
        df = df.head(limit)
    return df.to_dict("records")


def write_topk_csvs(scored: pd.DataFrame, out_dir: str, rank_by: str, topks: list[int]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ranked = scored[scored["status"] == "ok"].sort_values(rank_by, ascending=False)

    for topk in topks:
        top = ranked.head(topk)
        top[["prefix", "stained", "unstained"]].to_csv(
            os.path.join(out_dir, f"registered_clahe_tissue_top{topk}.csv"),
            index=False,
        )
        top.to_csv(
            os.path.join(out_dir, f"registered_clahe_tissue_top{topk}_scores.csv"),
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rank-by", default="ssim_content_gray")
    parser.add_argument("--topks", default="1000,1500,2000")
    args = parser.parse_args()

    rows = read_input(args.input, args.limit)
    if not rows:
        raise RuntimeError("No completed registered rows found to score.")

    print(f"Scoring rows: {len(rows)}")
    print(f"Workers: {args.workers}")
    print(f"Rank by: {args.rank_by}")
    print(f"Output: {args.out}")

    scored_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_row, row) for row in rows]
        for done, future in enumerate(as_completed(futures), 1):
            scored_rows.append(future.result())
            if done % 100 == 0 or done == len(futures):
                print(f"[{done:>5}/{len(futures):>5}]", flush=True)

    scored = pd.DataFrame(scored_rows)
    if args.rank_by not in scored.columns:
        raise ValueError(f"Unknown rank column: {args.rank_by}")

    scored = scored.sort_values(args.rank_by, ascending=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scored.to_csv(args.out, index=False)

    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]
    write_topk_csvs(scored, args.out_dir, args.rank_by, topks)

    ok = scored[scored["status"] == "ok"]
    print("Done")
    print(f"OK: {len(ok)}")
    print(f"Errors: {len(scored) - len(ok)}")
    print(f"Mean full RGB SSIM: {ok.ssim_full_rgb.mean():.4f}")
    print(f"Mean tissue gray SSIM: {ok.ssim_tissue_gray.mean():.4f}")
    print(f"Mean content gray SSIM: {ok.ssim_content_gray.mean():.4f}")
    print(f"Mean tissue fraction: {ok.tissue_fraction.mean():.4f}")
    print(f"Mean content fraction: {ok.content_fraction.mean():.4f}")


if __name__ == "__main__":
    main()
