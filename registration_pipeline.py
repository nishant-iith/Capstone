"""
TV-L1 Registration Pipeline — All Slides + Inline SSIM Scoring.
Workers register each pair AND compute post-registration SSIM in one pass.
Results sorted by SSIM descending. Top-K saved as training CSV.
"""

import os
import csv
import time
import re
import numpy as np
import multiprocessing as mp
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage import io, registration, transform
from skimage.metrics import structural_similarity as ssim_fn
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
STAINED_DIR   = "dataset/stained"
UNSTAINED_DIR = "dataset/unstained"
OUTPUT_DIR    = "data/processed/registered"
CSV_ALL       = "data/processed/registered_pairs_all.csv"   # all pairs + SSIM
CSV_TOPK      = "data/processed/registered_pairs.csv"       # top-K for training
NUM_WORKERS   = 60       # 64 cores, leave 4 for OS
FLOW_SIZE     = 1024     # full resolution TV-L1
TOP_K         = 2000     # keep top-K pairs by post-reg SSIM for training
LOAD_THREADS  = 48       # threads for image pre-loading

W = 68   # print width

# Global — populated in parent, inherited by forked workers via CoW
_ITEMS = []


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker_register(idx):
    """
    Register one pair and compute post-registration SSIM.
    Returns (idx, ssim_score, status_str).
    """
    item  = _ITEMS[idx]
    out_s = item["out_stained"]
    out_u = item["out_unstained"]

    # If already registered, load and score existing files
    if os.path.exists(out_s) and os.path.exists(out_u):
        try:
            s = io.imread(out_s)
            u = io.imread(out_u)
            sc = ssim_fn(s, u, channel_axis=2, data_range=255)
            return idx, sc, "skipped"
        except Exception:
            return idx, 0.0, "skipped"

    try:
        s_img = item["s_img"]
        u_img = item["u_img"]
        H, W  = s_img.shape[:2]

        # Grayscale for optical flow
        s_gray = cv2.cvtColor(s_img, cv2.COLOR_RGB2GRAY)
        u_gray = cv2.cvtColor(u_img, cv2.COLOR_RGB2GRAY)

        s_g = s_gray.astype(np.float32) / 255.0
        u_g = u_gray.astype(np.float32) / 255.0

        # TV-L1 optical flow at full resolution
        v_f, u_f = registration.optical_flow_tvl1(s_g, u_g)

        # Warp full-res unstained
        gy, gx = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32), indexing="ij"
        )
        warped = np.zeros_like(u_img, dtype=np.float64)
        for c in range(3):
            warped[:, :, c] = transform.warp(
                u_img[:, :, c] / 255.0,
                np.array([gy + v_f, gx + u_f]),
                mode="edge",
            )
        warped = (warped * 255).clip(0, 255).astype(np.uint8)

        # Save registered pair
        io.imsave(out_s, s_img,  check_contrast=False)
        io.imsave(out_u, warped, check_contrast=False)

        # Compute SSIM inline — no extra I/O pass needed
        sc = ssim_fn(s_img, warped, channel_axis=2, data_range=255)
        return idx, sc, "ok"

    except Exception as e:
        return idx, 0.0, f"error: {e}"


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_registration_pipeline():
    global _ITEMS

    print()
    print(f"  {'='*W}")
    print(f"  {'TV-L1 REGISTRATION PIPELINE  --  ALL SLIDES':^{W}}")
    print(f"  {'='*W}")
    print(f"  {'Method':<18}  TV-L1 Optical Flow  |  flow @ {FLOW_SIZE}px (full res)")
    print(f"  {'Workers':<18}  {NUM_WORKERS} fork-CoW processes  |  {LOAD_THREADS} load threads")
    print(f"  {'Input':<18}  {STAINED_DIR}  /  {UNSTAINED_DIR}")
    print(f"  {'Output':<18}  {OUTPUT_DIR}")
    print(f"  {'SSIM scoring':<18}  inline per-pair during registration (no 2nd pass)")
    print(f"  {'Top-K filter':<18}  keep top {TOP_K} pairs by post-reg SSIM -> training CSV")
    print(f"  {'='*W}")
    print()

    out_s_dir = os.path.join(OUTPUT_DIR, "stained")
    out_u_dir = os.path.join(OUTPUT_DIR, "unstained")
    os.makedirs(out_s_dir, exist_ok=True)
    os.makedirs(out_u_dir, exist_ok=True)

    # ── Build full pair list (ALL slides) ─────────────────────────────────────
    s_files = sorted(f for f in os.listdir(STAINED_DIR) if f.endswith("_stained.tif"))
    all_items, missing = [], 0
    for sf in s_files:
        uf = sf.replace("_stained.tif", "_unstained.tif")
        u_path = os.path.join(UNSTAINED_DIR, uf)
        if not os.path.exists(u_path):
            missing += 1
            continue
        prefix = sf.replace("_stained.tif", "")
        all_items.append({
            "prefix":         prefix,
            "stained_path":   os.path.join(STAINED_DIR, sf),
            "unstained_path": u_path,
            "out_stained":    os.path.join(out_s_dir, sf),
            "out_unstained":  os.path.join(out_u_dir, uf),
        })

    # Per-slide counts
    slide_counts = {}
    for p in all_items:
        m = re.search(r'(AS-\d+-\d+-Z\d+)', p["prefix"])
        s = m.group(1) if m else "unknown"
        slide_counts[s] = slide_counts.get(s, 0) + 1

    print(f"  Found {len(all_items)} pairs across {len(slide_counts)} slides:")
    for slide, cnt in sorted(slide_counts.items()):
        print(f"    {slide:<26}  {cnt:>5} patches")
    if missing:
        print(f"  WARNING: {missing} stained files have no matching unstained pair")
    print()

    already_done = sum(1 for p in all_items
                       if os.path.exists(p["out_stained"]) and os.path.exists(p["out_unstained"]))
    to_do = [p for p in all_items
             if not (os.path.exists(p["out_stained"]) and os.path.exists(p["out_unstained"]))]

    print(f"  Status: {already_done} already registered  |  {len(to_do)} to process")
    print()

    # ── Step 1: Load all images to RAM ────────────────────────────────────────
    print(f"  {'='*W}")
    print(f"  STEP 1 / 3  --  Loading {len(to_do)*2} images to RAM ({LOAD_THREADS} threads)")
    print(f"  {'='*W}")
    t0 = time.time()

    def _load(item):
        item["s_img"] = io.imread(item["stained_path"])
        item["u_img"] = io.imread(item["unstained_path"])
        return item

    # Only load the to_do items to RAM
    with ThreadPoolExecutor(max_workers=LOAD_THREADS) as pool:
        futs = {pool.submit(_load, item): item for item in to_do}
        n = 0
        for fut in as_completed(futs):
            fut.result()
            n += 1
            if n % 1000 == 0 or n == len(to_do):
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed > 0 else 0
                pct  = n / len(to_do) * 100 if to_do else 100
                bar  = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
                print(f"  [{bar}] {n:>5}/{len(to_do)}  {rate:.0f} img/s  {elapsed:.1f}s")

    _ITEMS.clear()
    _ITEMS.extend(all_items)
    print(f"  RAM loading complete. {len(to_do)*2} images cached | {time.time()-t0:.1f}s\n")

    # ── Step 2: Register + SSIM scoring ───────────────────────────────────────
    print(f"  {'='*W}")
    print(f"  STEP 2 / 3  --  Registering/Scoring {len(_ITEMS)} pairs")
    print(f"  {'='*W}")
    t1 = time.time()
    ctx = mp.get_context("fork")

    new_results = []
    # Try to load existing results to resume SSIM scoring
    if os.path.exists(CSV_ALL):
        try:
            with open(CSV_ALL, "r") as f:
                reader = csv.DictReader(f)
                new_results = [row for row in reader]
                for r in new_results:
                    r["ssim"] = float(r["ssim"])
            print(f"  Loaded {len(new_results)} existing results from {CSV_ALL}")
        except Exception as e:
            print(f"  Could not load existing CSV: {e}")

    errors      = []
    done        = 0
    ssim_running = []
    recent_ssims = collections.deque(maxlen=NUM_WORKERS)

    with ctx.Pool(processes=NUM_WORKERS) as pool:
        for idx, sc, status in pool.imap_unordered(
            _worker_register, range(len(_ITEMS)), chunksize=4
        ):
            done += 1
            item = _ITEMS[idx]
            if "error" not in status:
                res = {
                    "prefix":    item["prefix"],
                    "stained":   item["out_stained"],
                    "unstained": item["out_unstained"],
                    "ssim":      round(sc, 4),
                }
                new_results.append(res)
                ssim_running.append(sc)
                recent_ssims.append(sc)
            else:
                errors.append((item["prefix"], status))

            if done % 25 == 0 or done == len(_ITEMS):
                elapsed = time.time() - t1
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (len(_ITEMS) - done) / rate if rate > 0 else 0
                
                # Batch stats (last NUM_WORKERS)
                b_mean = np.mean(recent_ssims) if recent_ssims else 0
                b_min  = np.min(recent_ssims) if recent_ssims else 0
                b_max  = np.max(recent_ssims) if recent_ssims else 0
                
                pct     = done / len(_ITEMS) * 100
                bar     = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
                ts      = time.strftime("%H:%M:%S")
                print(f"  [{ts}] [{bar}] {done:>5}/{len(_ITEMS)}"
                      f"  {rate:.1f} p/s"
                      f"  ETA {eta/60:.1f}m"
                      f"  BATCH SSIM mean:{b_mean:.4f} min:{b_min:.4f} max:{b_max:.4f}"
                      f"  errors: {len(errors)}")

            # Save progress every 600 pairs
            if done % 600 == 0:
                with open(CSV_ALL, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["prefix", "stained", "unstained", "ssim"])
                    writer.writeheader()
                    writer.writerows(new_results)

    elapsed_reg = time.time() - t1
    print(f"\n  Registration done: {len(new_results)} ok  |  {len(errors)} errors  |  {elapsed_reg/60:.1f} min")
    if errors:
        print(f"  Sample errors: {errors[:3]}")
    print()

    # ── Step 3: Merge, sort by SSIM, write CSVs ───────────────────────────────
    print(f"  {'='*W}")
    print(f"  STEP 3 / 3  --  Sorting by SSIM + writing CSVs")
    print(f"  {'='*W}")

    # Collect previously skipped pairs that were scored in this run
    all_scored = new_results[:]

    # Load SSIM for already-registered pairs not in this run
    # (they were scored inline above when status="skipped")
    # new_results already contains skipped pairs with their SSIM

    # Sort all by SSIM descending
    all_scored.sort(key=lambda x: x["ssim"], reverse=True)

    # Write full CSV (all pairs + SSIM)
    os.makedirs(os.path.dirname(CSV_ALL), exist_ok=True)
    with open(CSV_ALL, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix", "stained", "unstained", "ssim"])
        writer.writeheader()
        writer.writerows(all_scored)

    # Write top-K training CSV (no SSIM column — matches existing training code)
    top_k = all_scored[:TOP_K]
    with open(CSV_TOPK, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix", "stained", "unstained"])
        writer.writeheader()
        for row in top_k:
            writer.writerow({k: row[k] for k in ["prefix", "stained", "unstained"]})

    # Stats
    ssim_all  = [r["ssim"] for r in all_scored]
    ssim_topk = [r["ssim"] for r in top_k]

    # Per-slide breakdown in top-K
    slide_topk = {}
    for r in top_k:
        m = re.search(r'(AS-\d+-\d+-Z\d+)', r["prefix"])
        s = m.group(1) if m else "unknown"
        slide_topk[s] = slide_topk.get(s, 0) + 1

    total_elapsed = time.time() - t0
    print()
    print(f"  {'='*W}")
    print(f"  {'REGISTRATION COMPLETE':^{W}}")
    print(f"  {'='*W}")
    print(f"  {'Total pairs registered':<30}  {len(all_scored)}")
    print(f"  {'Errors':<30}  {len(errors)}")
    print(f"  {'Total time':<30}  {total_elapsed/60:.1f} min")
    print(f"  {'-'*W}")
    print(f"  {'ALL PAIRS SSIM':<30}  mean={np.mean(ssim_all):.4f}  "
          f"min={np.min(ssim_all):.4f}  max={np.max(ssim_all):.4f}")
    print(f"  {'TOP-{TOP_K} SSIM':<30}  mean={np.mean(ssim_topk):.4f}  "
          f"min={np.min(ssim_topk):.4f}  max={np.max(ssim_topk):.4f}")
    print(f"  {'-'*W}")
    print(f"  Top-{TOP_K} slide breakdown:")
    for slide, cnt in sorted(slide_topk.items(), key=lambda x: -x[1]):
        print(f"    {slide:<26}  {cnt:>5} pairs in top-{TOP_K}")
    print(f"  {'-'*W}")
    print(f"  Full CSV  ({len(all_scored)} rows)  ->  {CSV_ALL}")
    print(f"  Train CSV ({len(top_k)} rows)  ->  {CSV_TOPK}")
    print(f"  {'='*W}")
    print()


if __name__ == "__main__":
    run_registration_pipeline()
