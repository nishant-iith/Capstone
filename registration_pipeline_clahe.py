"""
CLAHE TV-L1 registration pipeline.

This is a non-destructive replacement candidate for registration_pipeline.py.
It writes to data/processed/registered_clahe and separate CSVs, so the current
gray-TV-L1 registered dataset remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from skimage import io, registration, transform
from skimage.metrics import structural_similarity as ssim_fn


STAINED_DIR = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
OUTPUT_DIR = "data/processed/registered_clahe"
CSV_ALL = "data/processed/registered_clahe_pairs_all.csv"
CSV_TOPK = "data/processed/registered_clahe_pairs.csv"
OLD_CSV_ALL = "data/processed/registered_pairs_all.csv"
LIVE_LOG = "logs/registration_clahe_per_pair.csv"
PROGRESS_LOG = "logs/registration_clahe_progress.log"
TOP_K = 2000
DEFAULT_WORKERS = 8


def build_items(limit: int | None = None) -> list[dict[str, str]]:
    stained_files = sorted(f for f in os.listdir(STAINED_DIR) if f.endswith("_stained.tif"))
    items = []

    out_s_dir = os.path.join(OUTPUT_DIR, "stained")
    out_u_dir = os.path.join(OUTPUT_DIR, "unstained")
    os.makedirs(out_s_dir, exist_ok=True)
    os.makedirs(out_u_dir, exist_ok=True)

    for stained_file in stained_files:
        unstained_file = stained_file.replace("_stained.tif", "_unstained.tif")
        unstained_path = os.path.join(UNSTAINED_DIR, unstained_file)
        if not os.path.exists(unstained_path):
            continue

        prefix = stained_file.replace("_stained.tif", "")
        items.append(
            {
                "prefix": prefix,
                "stained_path": os.path.join(STAINED_DIR, stained_file),
                "unstained_path": unstained_path,
                "out_stained": os.path.join(out_s_dir, stained_file),
                "out_unstained": os.path.join(out_u_dir, unstained_file),
            }
        )

    if limit is not None:
        items = items[:limit]
    return items


def clahe_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced.astype(np.float32) / 255.0


def warp_rgb(moving: np.ndarray, v_flow: np.ndarray, u_flow: np.ndarray) -> np.ndarray:
    height, width = moving.shape[:2]
    gy, gx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    coords = np.array([gy + v_flow, gx + u_flow])

    warped = np.zeros_like(moving, dtype=np.float32)
    for channel in range(3):
        warped[:, :, channel] = transform.warp(
            moving[:, :, channel].astype(np.float32) / 255.0,
            coords,
            mode="edge",
            preserve_range=False,
        )
    return (warped * 255.0).clip(0, 255).astype(np.uint8)


def register_one(item: dict[str, str]) -> dict[str, object]:
    prefix = item["prefix"]
    out_stained = item["out_stained"]
    out_unstained = item["out_unstained"]

    try:
        if os.path.exists(out_stained) and os.path.exists(out_unstained):
            stained = io.imread(out_stained)
            warped = io.imread(out_unstained)
            score = ssim_fn(stained, warped, channel_axis=2, data_range=255)
            return {
                "prefix": prefix,
                "stained": out_stained,
                "unstained": out_unstained,
                "ssim": round(float(score), 4),
                "status": "skipped",
            }

        stained = io.imread(item["stained_path"])
        unstained = io.imread(item["unstained_path"])

        fixed = clahe_gray(stained)
        moving = clahe_gray(unstained)
        v_flow, u_flow = registration.optical_flow_tvl1(
            fixed,
            moving,
            attachment=15,
            tightness=0.3,
            num_warp=5,
            num_iter=10,
            prefilter=False,
            dtype=np.float32,
        )
        warped = warp_rgb(unstained, v_flow, u_flow)

        io.imsave(out_stained, stained, check_contrast=False)
        io.imsave(out_unstained, warped, check_contrast=False)

        score = ssim_fn(stained, warped, channel_axis=2, data_range=255)
        return {
            "prefix": prefix,
            "stained": out_stained,
            "unstained": out_unstained,
            "ssim": round(float(score), 4),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "prefix": prefix,
            "stained": out_stained,
            "unstained": out_unstained,
            "ssim": 0.0,
            "status": f"error: {exc}",
        }


def write_csvs(results: list[dict[str, object]], csv_all: str, csv_topk: str, top_k: int) -> None:
    os.makedirs(os.path.dirname(csv_all), exist_ok=True)
    scored = [row for row in results if not str(row["status"]).startswith("error")]
    scored.sort(key=lambda row: float(row["ssim"]), reverse=True)

    with open(csv_all, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix", "stained", "unstained", "ssim"])
        writer.writeheader()
        for row in scored:
            writer.writerow({key: row[key] for key in ["prefix", "stained", "unstained", "ssim"]})

    with open(csv_topk, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix", "stained", "unstained"])
        writer.writeheader()
        for row in scored[:top_k]:
            writer.writerow({key: row[key] for key in ["prefix", "stained", "unstained"]})


def load_old_scores(path: str) -> dict[str, float]:
    if not path or not os.path.exists(path):
        return {}

    scores: dict[str, float] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                scores[str(row["prefix"])] = float(row["ssim"])
            except (KeyError, TypeError, ValueError):
                continue
    return scores


def log_line(message: str, progress_handle) -> None:
    print(message, flush=True)
    if progress_handle is not None:
        progress_handle.write(message + "\n")
        progress_handle.flush()


def slide_counts(items: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        match = re.search(r"(AS-\d+-\d+-Z\d+)", item["prefix"])
        slide = match.group(1) if match else "unknown"
        counts[slide] = counts.get(slide, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--csv-all", default=CSV_ALL)
    parser.add_argument("--csv-topk", default=CSV_TOPK)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--old-csv", default=OLD_CSV_ALL)
    parser.add_argument("--live-log", default=LIVE_LOG)
    parser.add_argument("--progress-log", default=PROGRESS_LOG)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.live_log), exist_ok=True)
    os.makedirs(os.path.dirname(args.progress_log), exist_ok=True)

    items = build_items(limit=args.limit)
    counts = slide_counts(items)
    old_scores = load_old_scores(args.old_csv)

    start = time.time()
    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    live_fields = [
        "done",
        "total",
        "elapsed_sec",
        "prefix",
        "ssim",
        "old_ssim",
        "delta_vs_old",
        "status",
        "stained",
        "unstained",
    ]

    with open(args.progress_log, "w") as progress_f, open(args.live_log, "w", newline="") as live_f:
        live_writer = csv.DictWriter(live_f, fieldnames=live_fields)
        live_writer.writeheader()
        live_f.flush()

        log_line("CLAHE TV-L1 registration", progress_f)
        log_line(f"Pairs: {len(items)}", progress_f)
        log_line(f"Workers: {args.workers}", progress_f)
        log_line(f"Output: {OUTPUT_DIR}", progress_f)
        log_line(f"CSV all: {args.csv_all}", progress_f)
        log_line(f"CSV top-k: {args.csv_topk}", progress_f)
        log_line(f"Live per-pair log: {args.live_log}", progress_f)
        log_line(f"Progress log: {args.progress_log}", progress_f)
        log_line(f"Old-score comparison rows loaded: {len(old_scores)}", progress_f)
        log_line("Slides:", progress_f)
        for slide, count in sorted(counts.items()):
            log_line(f"  {slide:<24} {count:>5}", progress_f)

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(register_one, item): item for item in items}
            for done, future in enumerate(as_completed(futures), 1):
                row = future.result()
                old_ssim = old_scores.get(str(row["prefix"]))
                if old_ssim is None:
                    row["old_ssim"] = ""
                    row["delta_vs_old"] = ""
                else:
                    row["old_ssim"] = round(old_ssim, 4)
                    row["delta_vs_old"] = round(float(row["ssim"]) - old_ssim, 4)

                results.append(row)
                if str(row["status"]).startswith("error"):
                    errors.append(row)

                live_writer.writerow(
                    {
                        "done": done,
                        "total": len(items),
                        "elapsed_sec": round(time.time() - start, 2),
                        "prefix": row["prefix"],
                        "ssim": row["ssim"],
                        "old_ssim": row["old_ssim"],
                        "delta_vs_old": row["delta_vs_old"],
                        "status": row["status"],
                        "stained": row["stained"],
                        "unstained": row["unstained"],
                    }
                )
                live_f.flush()

                if done % 25 == 0 or done == len(items):
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (len(items) - done) / rate if rate > 0 else 0.0
                    recent = [float(r["ssim"]) for r in results[-min(100, len(results)):]]
                    deltas = [
                        float(r["delta_vs_old"])
                        for r in results[-min(100, len(results)):]
                        if r["delta_vs_old"] != ""
                    ]
                    delta_msg = f" recent_delta mean={np.mean(deltas):.4f}" if deltas else ""
                    log_line(
                        f"[{done:>5}/{len(items):>5}] "
                        f"{rate:5.2f} pairs/s ETA {eta/60:5.1f}m "
                        f"recent_ssim mean={np.mean(recent):.4f}"
                        f"{delta_msg} "
                        f"errors={len(errors)}",
                        progress_f,
                    )

                if done % args.save_every == 0:
                    write_csvs(results, args.csv_all, args.csv_topk, args.top_k)

        write_csvs(results, args.csv_all, args.csv_topk, args.top_k)

        scored = [float(row["ssim"]) for row in results if not str(row["status"]).startswith("error")]
        log_line("Done", progress_f)
        log_line(f"OK: {len(scored)}", progress_f)
        log_line(f"Errors: {len(errors)}", progress_f)
        if scored:
            log_line(
                f"SSIM mean={np.mean(scored):.4f} "
                f"min={np.min(scored):.4f} "
                f"max={np.max(scored):.4f}",
                progress_f,
            )
        deltas = [float(row["delta_vs_old"]) for row in results if row.get("delta_vs_old") != ""]
        if deltas:
            log_line(
                f"Delta vs old mean={np.mean(deltas):.4f} "
                f"min={np.min(deltas):.4f} "
                f"max={np.max(deltas):.4f}",
                progress_f,
            )
        log_line(f"Elapsed: {(time.time() - start) / 60:.1f} min", progress_f)


if __name__ == "__main__":
    main()
