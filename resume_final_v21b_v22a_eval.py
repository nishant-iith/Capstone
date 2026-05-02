"""
Resume and finalize eval_v21b_v22a_final.py.

The original final eval can be interrupted while writing the per-pair CSV. This
script reads completed (dataset, prefix, method) rows, evaluates only missing
rows, appends them, then writes the final summary.
"""

import csv
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import eval_v21b_v22a_final as final_eval


DIRECT_METHODS = [
    "v20_base",
    "v20_tta4",
    "v22_base",
    "v22_tta4",
    "v21b_base",
    "v21b_tta4",
]
PAIR_20_22 = [f"ens20_22_w22_{w:.2f}" for w in final_eval.PAIR_WEIGHTS]
PAIR_22_21B = [f"ens22_21b_w21b_{w:.2f}" for w in final_eval.PAIR_WEIGHTS]
TRIPLE = [f"ens3_{w20:.2f}_{w22:.2f}_{w21b:.2f}" for w20, w22, w21b in final_eval.TRIPLE_WEIGHTS]
ALL_METHODS = DIRECT_METHODS + PAIR_20_22 + PAIR_22_21B + TRIPLE


def completed_keys():
    if not os.path.exists(final_eval.OUT_CSV):
        return set()
    df = pd.read_csv(final_eval.OUT_CSV)
    return set(zip(df["dataset"], df["prefix"], df["method"]))


def prediction_map(v20, v22, v21b, unstained):
    p20_base = final_eval.to_np(final_eval.predict_base(v20, unstained))
    p22_base = final_eval.to_np(final_eval.predict_base(v22, unstained))
    p21b_base = final_eval.to_np(final_eval.predict_base(v21b, unstained))
    p20 = final_eval.to_np(final_eval.predict_tta4(v20, unstained))
    p22 = final_eval.to_np(final_eval.predict_tta4(v22, unstained))
    p21b = final_eval.to_np(final_eval.predict_tta4(v21b, unstained))

    preds = {
        "v20_base": p20_base,
        "v20_tta4": p20,
        "v22_base": p22_base,
        "v22_tta4": p22,
        "v21b_base": p21b_base,
        "v21b_tta4": p21b,
    }
    for w22 in final_eval.PAIR_WEIGHTS:
        preds[f"ens20_22_w22_{w22:.2f}"] = np.clip((1.0 - w22) * p20 + w22 * p22, 0.0, 1.0)
    for w21b in final_eval.PAIR_WEIGHTS:
        preds[f"ens22_21b_w21b_{w21b:.2f}"] = np.clip((1.0 - w21b) * p22 + w21b * p21b, 0.0, 1.0)
    for w20, w22, w21b in final_eval.TRIPLE_WEIGHTS:
        preds[f"ens3_{w20:.2f}_{w22:.2f}_{w21b:.2f}"] = np.clip(
            w20 * p20 + w22 * p22 + w21b * p21b, 0.0, 1.0
        )
    return preds


def write_summary():
    df = pd.read_csv(final_eval.OUT_CSV)
    df = df.drop_duplicates(["dataset", "prefix", "method"], keep="last")
    lines = ["Final v20/v22A/v21B fixed evaluation", ""]
    for dataset_name in ("old_registered", "clahe_same_prefixes"):
        g = df[df["dataset"] == dataset_name]
        counts = g.groupby("method")["prefix"].nunique()
        if counts.empty:
            continue
        expected = int(counts.max())
        incomplete = counts[counts < expected]
        if not incomplete.empty:
            lines.append(f"WARNING: {dataset_name} has incomplete methods: {incomplete.to_dict()}")
            lines.append("")
        means = (
            g.groupby("method")[["ssim", "psnr", "pcc"]]
            .mean()
            .sort_values(["ssim", "psnr", "pcc"], ascending=False)
        )
        lines.append(f"{dataset_name} ({g['prefix'].nunique()} prefixes)")
        lines.append(f"{'rank':>4} {'method':<22} {'SSIM':>8} {'PSNR':>8} {'PCC':>8}")
        lines.append("-" * 58)
        for rank, (method, row) in enumerate(means.iterrows(), start=1):
            lines.append(
                f"{rank:4d} {method:<22} {row['ssim']:8.4f} {row['psnr']:8.2f} {row['pcc']:8.4f}"
            )
        lines.append("")
    summary = "\n".join(lines).rstrip()
    with open(final_eval.OUT_SUMMARY, "w") as f:
        f.write(summary + "\n")
    print(summary)
    print(f"\nWrote {final_eval.OUT_SUMMARY}")


def main():
    rows_by_dataset = final_eval.fixed_rows()
    done = completed_keys()
    missing_total = 0
    for dataset_name, rows in rows_by_dataset.items():
        for prefix in rows["prefix"]:
            missing_total += sum((dataset_name, prefix, method) not in done for method in ALL_METHODS)
    print(f"Missing rows to evaluate: {missing_total}")
    if missing_total == 0:
        write_summary()
        return

    print(f"Loading models on {final_eval.DEVICE}...")
    v20 = final_eval.load_v20_style(final_eval.V20_PATH)
    v22 = final_eval.load_v20_style(final_eval.V22_PATH)
    v21b = final_eval.load_v21b(final_eval.V21B_PATH)

    t0 = time.time()
    wrote = 0
    with open(final_eval.OUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for dataset_name, rows in rows_by_dataset.items():
            ds = final_eval.FixedRowsDataset(rows, dataset_name)
            loader = DataLoader(
                ds,
                batch_size=final_eval.BATCH_SIZE,
                shuffle=False,
                num_workers=final_eval.NUM_WORKERS,
                pin_memory=True,
                persistent_workers=final_eval.NUM_WORKERS > 0,
            )
            for idx, (_, prefixes, unstained, stained) in enumerate(loader, start=1):
                prefix = prefixes[0]
                missing_methods = [
                    method for method in ALL_METHODS if (dataset_name, prefix, method) not in done
                ]
                if not missing_methods:
                    continue

                unstained = unstained.to(final_eval.DEVICE, non_blocking=True)
                target_np = final_eval.to_np(stained)
                preds = prediction_map(v20, v22, v21b, unstained)
                for method in missing_methods:
                    ssim, psnr, pcc = final_eval.metrics(preds[method], target_np)
                    writer.writerow(
                        [dataset_name, prefix, method, f"{ssim:.6f}", f"{psnr:.6f}", f"{pcc:.6f}"]
                    )
                    done.add((dataset_name, prefix, method))
                    wrote += 1
                f.flush()
                print(
                    f"{dataset_name} {idx:3d}/{len(ds)} wrote {len(missing_methods)} missing rows "
                    f"({wrote}/{missing_total})",
                    flush=True,
                )

    print(f"Resume completed in {time.time() - t0:.1f}s. Appended {wrote} rows.")
    write_summary()


if __name__ == "__main__":
    main()
