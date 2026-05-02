"""
Create training CSV variants from CLAHE registration logs.

Outputs plain top-K CLAHE CSVs plus best-of-old-vs-CLAHE CSVs that fall back to
the old registered pair when CLAHE reduced SSIM for a specific patch.
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd


LIVE_LOG = "logs/registration_clahe_per_pair.csv"
OLD_CSV = "data/processed/registered_pairs_all.csv"
OUT_DIR = "data/processed/training_csv_variants"


def add_slide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["slide"] = df["prefix"].str.extract(r"(AS-\d+-\d+-Z\d+)")[0]
    return df


def write_training_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[["prefix", "stained", "unstained"]].to_csv(path, index=False)


def summarize(df: pd.DataFrame, score_col: str, name: str) -> dict[str, object]:
    top = add_slide(df)
    return {
        "name": name,
        "rows": len(top),
        "score_mean": float(top[score_col].mean()),
        "score_min": float(top[score_col].min()),
        "score_max": float(top[score_col].max()),
        "slides": int(top["slide"].nunique()),
        "negative_delta_rows": int((top.get("delta_vs_old", pd.Series(dtype=float)) < 0).sum()),
        "slide_counts": top["slide"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-log", default=LIVE_LOG)
    parser.add_argument("--old-csv", default=OLD_CSV)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--topks", default="1000,1500,2000")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]

    live = pd.read_csv(args.live_log)
    scored = live[~live["status"].astype(str).str.startswith("error")].copy()
    scored = scored.drop_duplicates("prefix", keep="last")
    scored["ssim"] = scored["ssim"].astype(float)
    scored["old_ssim"] = scored["old_ssim"].astype(float)
    scored["delta_vs_old"] = scored["delta_vs_old"].astype(float)

    old = pd.read_csv(args.old_csv).rename(
        columns={
            "stained": "old_stained",
            "unstained": "old_unstained",
            "ssim": "old_csv_ssim",
        }
    )
    merged = scored.merge(
        old[["prefix", "old_stained", "old_unstained", "old_csv_ssim"]],
        on="prefix",
        how="left",
    )

    use_clahe = merged["ssim"] >= merged["old_ssim"]
    best = merged.copy()
    best["best_source"] = use_clahe.map({True: "clahe", False: "old"})
    best["best_ssim"] = merged["ssim"].where(use_clahe, merged["old_ssim"])
    best["stained"] = merged["stained"].where(use_clahe, merged["old_stained"])
    best["unstained"] = merged["unstained"].where(use_clahe, merged["old_unstained"])

    summaries: list[dict[str, object]] = []

    for topk in topks:
        clahe_top = scored.sort_values("ssim", ascending=False).head(topk)
        clahe_name = f"registered_clahe_top{topk}"
        write_training_csv(clahe_top, os.path.join(args.out_dir, f"{clahe_name}.csv"))
        clahe_top.to_csv(os.path.join(args.out_dir, f"{clahe_name}_scores.csv"), index=False)
        summaries.append(summarize(clahe_top, "ssim", clahe_name))

        positive_top = (
            scored[scored["delta_vs_old"] >= 0]
            .sort_values("ssim", ascending=False)
            .head(topk)
        )
        positive_name = f"registered_clahe_positive_top{topk}"
        write_training_csv(positive_top, os.path.join(args.out_dir, f"{positive_name}.csv"))
        positive_top.to_csv(os.path.join(args.out_dir, f"{positive_name}_scores.csv"), index=False)
        summaries.append(summarize(positive_top, "ssim", positive_name))

        best_top = best.sort_values("best_ssim", ascending=False).head(topk)
        best_name = f"registered_bestof_old_clahe_top{topk}"
        write_training_csv(best_top, os.path.join(args.out_dir, f"{best_name}.csv"))
        best_top.to_csv(os.path.join(args.out_dir, f"{best_name}_scores.csv"), index=False)
        summaries.append(summarize(best_top, "best_ssim", best_name))

    rows = []
    for summary in summaries:
        slide_counts = summary.pop("slide_counts")
        rows.append(summary | {"slide_counts": "; ".join(f"{k}:{v}" for k, v in slide_counts.items())})

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(summary_df[["name", "rows", "score_mean", "score_min", "score_max", "slides", "negative_delta_rows"]].to_string(index=False))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
