"""
Create high-content + high-registration-quality training CSVs.

Ranks registered CLAHE pairs by a percentile-combined score:
  45% content-region SSIM, 30% full RGB SSIM, 20% content fraction,
   5% positive gain over old registration.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


SCORES = "data/processed/registered_clahe_content_scores.csv"
OUT_DIR = "data/processed/content_quality_csvs"


def add_slide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["slide"] = df["prefix"].str.extract(r"(AS-\d+-\d+-Z\d+)")[0]
    return df


def rank_pct(series: pd.Series) -> pd.Series:
    return series.astype(float).rank(method="average", pct=True)


def write_training_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[["prefix", "stained", "unstained"]].to_csv(path, index=False)


def summarize(df: pd.DataFrame, name: str) -> dict[str, object]:
    top = add_slide(df)
    return {
        "name": name,
        "rows": len(top),
        "combined_mean": top["content_quality_score"].mean(),
        "full_rgb_mean": top["ssim_full_rgb"].mean(),
        "content_gray_mean": top["ssim_content_gray"].mean(),
        "content_fraction_mean": top["content_fraction"].mean(),
        "delta_mean": top["delta_vs_old"].mean(),
        "delta_min": top["delta_vs_old"].min(),
        "slides": top["slide"].nunique(),
        "negative_delta_rows": int((top["delta_vs_old"] < 0).sum()),
        "slide_counts": top["slide"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--topks", default="1000,1500,2000")
    parser.add_argument("--min-full-rgb", type=float, default=0.50)
    parser.add_argument("--require-positive-delta", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]

    df = pd.read_csv(args.scores)
    df = df[df["status"] == "ok"].copy()
    numeric_cols = [
        "ssim_full_rgb",
        "ssim_content_gray",
        "content_fraction",
        "delta_vs_old",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    # Percentile ranks make the weighted score robust to different metric scales.
    df["rank_content_gray"] = rank_pct(df["ssim_content_gray"])
    df["rank_full_rgb"] = rank_pct(df["ssim_full_rgb"])
    df["rank_content_fraction"] = rank_pct(df["content_fraction"])
    df["rank_delta"] = rank_pct(df["delta_vs_old"].clip(lower=0.0))
    df["content_quality_score"] = (
        0.45 * df["rank_content_gray"]
        + 0.30 * df["rank_full_rgb"]
        + 0.20 * df["rank_content_fraction"]
        + 0.05 * df["rank_delta"]
    )

    df.to_csv(os.path.join(args.out_dir, "registered_clahe_content_quality_all_scores.csv"), index=False)

    filtered = df[df["ssim_full_rgb"] >= args.min_full_rgb].copy()
    if args.require_positive_delta:
        filtered = filtered[filtered["delta_vs_old"] >= 0].copy()

    summaries: list[dict[str, object]] = []
    for topk in topks:
        top = df.sort_values("content_quality_score", ascending=False).head(topk)
        name = f"content_quality_top{topk}"
        write_training_csv(top, os.path.join(args.out_dir, f"{name}.csv"))
        top.to_csv(os.path.join(args.out_dir, f"{name}_scores.csv"), index=False)
        summaries.append(summarize(top, name))

        filt_top = filtered.sort_values("content_quality_score", ascending=False).head(topk)
        filt_name = f"content_quality_minrgb{args.min_full_rgb:.2f}_positive_top{topk}"
        write_training_csv(filt_top, os.path.join(args.out_dir, f"{filt_name}.csv"))
        filt_top.to_csv(os.path.join(args.out_dir, f"{filt_name}_scores.csv"), index=False)
        summaries.append(summarize(filt_top, filt_name))

    summary_rows = []
    for summary in summaries:
        slide_counts = summary.pop("slide_counts")
        summary_rows.append(
            summary
            | {"slide_counts": "; ".join(f"{slide}:{count}" for slide, count in slide_counts.items())}
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(
        summary_df[
            [
                "name",
                "rows",
                "full_rgb_mean",
                "content_gray_mean",
                "content_fraction_mean",
                "delta_mean",
                "delta_min",
                "slides",
                "negative_delta_rows",
            ]
        ].to_string(index=False)
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
