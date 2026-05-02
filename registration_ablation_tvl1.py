"""
Controlled TV-L1 registration ablation.

This script does not modify the existing registered dataset. It samples pairs
from the current registration CSV, re-runs several TV-L1 variants on the raw
images, and writes per-method scores to logs/registration_ablation_tvl1.csv.

The goal is to prove whether a registration change improves the data before
spending CPU time on a full 8k+ pair re-registration.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from skimage import filters, io, registration, transform
from skimage.metrics import structural_similarity as ssim_fn


STAINED_DIR = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
CURRENT_CSV = "data/processed/registered_pairs_all.csv"
OUT_CSV = "logs/registration_ablation_tvl1.csv"
SUMMARY_TXT = "logs/registration_ablation_tvl1_summary.txt"


@dataclass(frozen=True)
class Method:
    name: str
    preprocess: str
    attachment: float = 15.0
    tightness: float = 0.3
    num_warp: int = 5
    num_iter: int = 10
    prefilter: bool = False


METHODS = [
    Method("gray_default", "gray"),
    Method("gray_prefilter", "gray", prefilter=True),
    Method("gray_more_iter", "gray", num_iter=20),
    Method("gray_smoother", "gray", attachment=10.0, tightness=0.4, num_iter=20),
    Method("clahe_default", "clahe"),
    Method("clahe_prefilter", "clahe", prefilter=True),
    Method("edge_default", "edge"),
    Method("mix_clahe_edge", "mix"),
]


def raw_paths(prefix: str) -> tuple[str, str]:
    return (
        os.path.join(STAINED_DIR, f"{prefix}_stained.tif"),
        os.path.join(UNSTAINED_DIR, f"{prefix}_unstained.tif"),
    )


def current_paths(prefix: str) -> tuple[str, str]:
    return (
        f"data/processed/registered/stained/{prefix}_stained.tif",
        f"data/processed/registered/unstained/{prefix}_unstained.tif",
    )


def to_gray_float(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def preprocess_pair(s_img: np.ndarray, u_img: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    s_gray = to_gray_float(s_img)
    u_gray = to_gray_float(u_img)

    if mode == "gray":
        return s_gray, u_gray

    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        s = clahe.apply((s_gray * 255).astype(np.uint8)).astype(np.float32) / 255.0
        u = clahe.apply((u_gray * 255).astype(np.uint8)).astype(np.float32) / 255.0
        return s, u

    if mode == "edge":
        return normalize01(filters.sobel(s_gray)), normalize01(filters.sobel(u_gray))

    if mode == "mix":
        s_edge = normalize01(filters.sobel(s_gray))
        u_edge = normalize01(filters.sobel(u_gray))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        s_clahe = clahe.apply((s_gray * 255).astype(np.uint8)).astype(np.float32) / 255.0
        u_clahe = clahe.apply((u_gray * 255).astype(np.uint8)).astype(np.float32) / 255.0
        return normalize01(0.7 * s_clahe + 0.3 * s_edge), normalize01(0.7 * u_clahe + 0.3 * u_edge)

    raise ValueError(f"Unknown preprocess mode: {mode}")


def warp_rgb(moving: np.ndarray, v_flow: np.ndarray, u_flow: np.ndarray) -> np.ndarray:
    height, width = moving.shape[:2]
    gy, gx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    warped = np.zeros_like(moving, dtype=np.float32)
    coords = np.array([gy + v_flow, gx + u_flow])
    for channel in range(3):
        warped[:, :, channel] = transform.warp(
            moving[:, :, channel].astype(np.float32) / 255.0,
            coords,
            mode="edge",
            preserve_range=False,
        )
    return (warped * 255.0).clip(0, 255).astype(np.uint8)


def score_pair(stained: np.ndarray, unstained: np.ndarray) -> tuple[float, float]:
    rgb = float(ssim_fn(stained, unstained, channel_axis=2, data_range=255))
    gray = float(ssim_fn(to_gray_float(stained), to_gray_float(unstained), data_range=1.0))
    return rgb, gray


def resize_for_flow(img: np.ndarray, flow_size: int) -> tuple[np.ndarray, float, float]:
    height, width = img.shape[:2]
    max_side = max(height, width)
    if flow_size <= 0 or max_side <= flow_size:
        return img, 1.0, 1.0

    scale = flow_size / float(max_side)
    new_width = max(16, int(round(width * scale)))
    new_height = max(16, int(round(height * scale)))
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, height / float(new_height), width / float(new_width)


def upsample_flow(
    v_flow: np.ndarray,
    u_flow: np.ndarray,
    out_shape: tuple[int, int],
    row_scale: float,
    col_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    out_height, out_width = out_shape
    if v_flow.shape == out_shape:
        return v_flow, u_flow

    v_full = cv2.resize(v_flow, (out_width, out_height), interpolation=cv2.INTER_LINEAR) * row_scale
    u_full = cv2.resize(u_flow, (out_width, out_height), interpolation=cv2.INTER_LINEAR) * col_scale
    return v_full.astype(np.float32), u_full.astype(np.float32)


def register_with_method(
    s_img: np.ndarray,
    u_img: np.ndarray,
    method: Method,
    flow_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    s_drive, u_drive = preprocess_pair(s_img, u_img, method.preprocess)
    s_flow, row_scale, col_scale = resize_for_flow(s_drive, flow_size)
    u_flow_img, _, _ = resize_for_flow(u_drive, flow_size)
    v_flow, u_flow = registration.optical_flow_tvl1(
        s_flow,
        u_flow_img,
        attachment=method.attachment,
        tightness=method.tightness,
        num_warp=method.num_warp,
        num_iter=method.num_iter,
        prefilter=method.prefilter,
        dtype=np.float32,
    )
    v_flow, u_flow = upsample_flow(v_flow, u_flow, s_img.shape[:2], row_scale, col_scale)
    warped = warp_rgb(u_img, v_flow, u_flow)
    stats = {
        "flow_abs_mean": float(np.mean(np.sqrt(v_flow * v_flow + u_flow * u_flow))),
        "flow_abs_p95": float(np.percentile(np.sqrt(v_flow * v_flow + u_flow * u_flow), 95)),
    }
    return warped, stats


def choose_samples(csv_path: str, samples_per_tier: int) -> list[dict[str, str]]:
    df = pd.read_csv(csv_path).sort_values("ssim", ascending=False).reset_index(drop=True)
    rows = []

    tiers: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
        ("high", lambda x: x.head(max(samples_per_tier * 4, samples_per_tier))),
        ("topk_cutoff", lambda x: x.iloc[max(0, 2000 - samples_per_tier * 2): 2000 + samples_per_tier * 2]),
        ("mid", lambda x: x.iloc[max(0, len(x) // 2 - samples_per_tier * 2): len(x) // 2 + samples_per_tier * 2]),
        ("low", lambda x: x.iloc[max(0, int(len(x) * 0.10) - samples_per_tier * 2): int(len(x) * 0.10) + samples_per_tier * 2]),
    ]

    for tier, selector in tiers:
        subset = selector(df)
        if subset.empty:
            continue
        picked = subset.sample(
            n=min(samples_per_tier, len(subset)),
            random_state=42,
        )
        for _, row in picked.iterrows():
            prefix = str(row["prefix"])
            s_path, u_path = raw_paths(prefix)
            reg_s, reg_u = current_paths(prefix)
            if all(os.path.exists(p) for p in [s_path, u_path, reg_s, reg_u]):
                rows.append(
                    {
                        "tier": tier,
                        "prefix": prefix,
                        "csv_current_ssim": float(row["ssim"]),
                    }
                )

    seen = set()
    unique_rows = []
    for row in rows:
        key = row["prefix"]
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def evaluate_prefix(
    row: dict[str, str],
    flow_size: int,
    methods: list[Method],
) -> list[dict[str, object]]:
    prefix = str(row["prefix"])
    tier = str(row["tier"])
    s_path, u_path = raw_paths(prefix)
    reg_s, reg_u = current_paths(prefix)

    s_img = io.imread(s_path)
    u_img = io.imread(u_path)
    reg_u_img = io.imread(reg_u)

    raw_rgb, raw_gray = score_pair(s_img, u_img)
    current_rgb, current_gray = score_pair(s_img, reg_u_img)

    records: list[dict[str, object]] = [
        {
            "tier": tier,
            "prefix": prefix,
            "method": "raw_unregistered",
            "rgb_ssim": raw_rgb,
            "gray_ssim": raw_gray,
            "delta_vs_current_rgb": raw_rgb - current_rgb,
            "delta_vs_raw_rgb": 0.0,
            "elapsed_sec": 0.0,
            "flow_abs_mean": 0.0,
            "flow_abs_p95": 0.0,
            "flow_size": flow_size,
            "csv_current_ssim": row["csv_current_ssim"],
            "current_saved_rgb_ssim": current_rgb,
            "current_saved_gray_ssim": current_gray,
        },
        {
            "tier": tier,
            "prefix": prefix,
            "method": "current_saved",
            "rgb_ssim": current_rgb,
            "gray_ssim": current_gray,
            "delta_vs_current_rgb": 0.0,
            "delta_vs_raw_rgb": current_rgb - raw_rgb,
            "elapsed_sec": 0.0,
            "flow_abs_mean": 0.0,
            "flow_abs_p95": 0.0,
            "flow_size": flow_size,
            "csv_current_ssim": row["csv_current_ssim"],
            "current_saved_rgb_ssim": current_rgb,
            "current_saved_gray_ssim": current_gray,
        },
    ]

    for method in methods:
        start = time.time()
        try:
            warped, stats = register_with_method(s_img, u_img, method, flow_size)
            rgb, gray = score_pair(s_img, warped)
            records.append(
                {
                    "tier": tier,
                    "prefix": prefix,
                    "method": method.name,
                    "rgb_ssim": rgb,
                    "gray_ssim": gray,
                    "delta_vs_current_rgb": rgb - current_rgb,
                    "delta_vs_raw_rgb": rgb - raw_rgb,
                    "elapsed_sec": time.time() - start,
                    "flow_abs_mean": stats["flow_abs_mean"],
                    "flow_abs_p95": stats["flow_abs_p95"],
                    "flow_size": flow_size,
                    "csv_current_ssim": row["csv_current_ssim"],
                    "current_saved_rgb_ssim": current_rgb,
                    "current_saved_gray_ssim": current_gray,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "tier": tier,
                    "prefix": prefix,
                    "method": method.name,
                    "rgb_ssim": 0.0,
                    "gray_ssim": 0.0,
                    "delta_vs_current_rgb": -current_rgb,
                    "delta_vs_raw_rgb": -raw_rgb,
                    "elapsed_sec": time.time() - start,
                    "flow_abs_mean": 0.0,
                    "flow_abs_p95": 0.0,
                    "flow_size": flow_size,
                    "csv_current_ssim": row["csv_current_ssim"],
                    "current_saved_rgb_ssim": current_rgb,
                    "current_saved_gray_ssim": current_gray,
                    "error": repr(exc),
                }
            )

    return records


def write_summary(df: pd.DataFrame, output_path: str) -> None:
    method_rows = []
    for method, group in df.groupby("method"):
        if method == "raw_unregistered":
            continue
        method_rows.append(
            {
                "method": method,
                "mean_rgb": group["rgb_ssim"].mean(),
                "mean_delta_vs_current": group["delta_vs_current_rgb"].mean(),
                "win_rate_vs_current": (group["delta_vs_current_rgb"] > 0).mean(),
                "median_elapsed_sec": group["elapsed_sec"].median(),
            }
        )
    summary = pd.DataFrame(method_rows).sort_values("mean_rgb", ascending=False)

    tier_summary = (
        df[df["method"] != "raw_unregistered"]
        .groupby(["tier", "method"])["rgb_ssim"]
        .mean()
        .reset_index()
        .sort_values(["tier", "rgb_ssim"], ascending=[True, False])
    )

    with open(output_path, "w") as f:
        f.write("TV-L1 registration ablation summary\n")
        f.write("===================================\n\n")
        f.write(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        f.write("\n\nPer-tier mean RGB SSIM\n")
        f.write(tier_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-tier", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--flow-size",
        type=int,
        default=512,
        help="Max side length used to compute TV-L1 flow. Use 1024 for full-resolution parity.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method names to run, or 'all'.",
    )
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--summary", default=SUMMARY_TXT)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    samples = choose_samples(CURRENT_CSV, args.samples_per_tier)
    if not samples:
        raise RuntimeError("No valid samples found for registration ablation.")

    if args.methods == "all":
        methods = METHODS
    else:
        requested = {name.strip() for name in args.methods.split(",") if name.strip()}
        methods = [method for method in METHODS if method.name in requested]
        missing = sorted(requested - {method.name for method in methods})
        if missing:
            raise ValueError(f"Unknown method(s): {', '.join(missing)}")
    if not methods:
        raise ValueError("No methods selected.")

    print(f"Selected {len(samples)} unique samples")
    print(f"Workers: {args.workers}")
    print(f"Flow size: {args.flow_size}")
    print(f"Methods: {', '.join(method.name for method in methods)}")
    print(f"Writing: {args.out_csv}")

    all_records: list[dict[str, object]] = []
    fieldnames = [
        "tier",
        "prefix",
        "method",
        "rgb_ssim",
        "gray_ssim",
        "delta_vs_current_rgb",
        "delta_vs_raw_rgb",
        "elapsed_sec",
        "flow_abs_mean",
        "flow_abs_p95",
        "flow_size",
        "csv_current_ssim",
        "current_saved_rgb_ssim",
        "current_saved_gray_ssim",
        "error",
    ]

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if args.workers == 1:
            for i, row in enumerate(samples, 1):
                records = evaluate_prefix(row, args.flow_size, methods)
                for record in records:
                    writer.writerow(record)
                f.flush()
                all_records.extend(records)
                print(f"[{i:03d}/{len(samples):03d}] {row['tier']} {row['prefix']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(evaluate_prefix, row, args.flow_size, methods): row
                    for row in samples
                }
                for i, fut in enumerate(as_completed(futures), 1):
                    row = futures[fut]
                    records = fut.result()
                    for record in records:
                        writer.writerow(record)
                    f.flush()
                    all_records.extend(records)
                    print(f"[{i:03d}/{len(samples):03d}] {row['tier']} {row['prefix']}", flush=True)

    df = pd.DataFrame(all_records)
    write_summary(df, args.summary)
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
