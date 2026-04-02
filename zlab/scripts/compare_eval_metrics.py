#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = ["mean_ic", "mean_rank_ic", "mae"]
SELECTION_ORDER = {"default": 0, "cross-entropy": 1, "rankIc": 2}
SPLIT_ORDER = {"val": 0, "test": 1}
GROUP_ORDER = {"group_a": 0, "group_b": 1, "group_c": 2}
METRIC_TITLES = {
    "mean_ic": "Mean IC",
    "mean_rank_ic": "Mean RankIC",
    "mae": "MAE",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment metrics.json files and generate comparison plots."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="zlab/results/evaluations",
        help="Root directory containing recursive evaluation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="zlab/results/evaluation_comparisons",
        help="Directory where summary tables and figures are written.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to compare. Default: mean_ic mean_rank_ic mae",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional split filter, e.g. test or val test.",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=None,
        help="Optional horizon filter, e.g. h1 h5.",
    )
    return parser.parse_args()


def infer_group(run_name: str) -> str:
    for prefix in GROUP_ORDER:
        if run_name.startswith(prefix):
            return prefix
    return "other"


def parse_metrics_path(metrics_path: Path, root: Path) -> dict | None:
    rel_parts = metrics_path.relative_to(root).parts
    if len(rel_parts) < 4 or rel_parts[-1] != "metrics.json":
        return None

    split = rel_parts[-2]
    if split not in SPLIT_ORDER:
        return None

    horizon = rel_parts[0]
    run_name = rel_parts[1]
    selection = "default" if len(rel_parts) == 4 else "/".join(rel_parts[2:-2])
    label = run_name if selection == "default" else f"{run_name}/{selection}"

    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    row = {
        "horizon": horizon,
        "split": split,
        "run_name": run_name,
        "selection": selection,
        "group": infer_group(run_name),
        "label": label,
        "metrics_path": str(metrics_path),
        "result_dir": str(metrics_path.parent),
    }
    row.update(payload)
    return row


def load_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        row = parse_metrics_path(metrics_path, root)
        if row is not None:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No metrics.json found under {root}")

    df = pd.DataFrame(rows)
    df["horizon_num"] = df["horizon"].str.extract(r"(\d+)").astype(float)
    df["group_order"] = df["group"].map(GROUP_ORDER).fillna(99)
    df["split_order"] = df["split"].map(SPLIT_ORDER).fillna(99)
    df["selection_order"] = df["selection"].map(SELECTION_ORDER).fillna(99)
    return df.sort_values(
        ["horizon_num", "split_order", "group_order", "run_name", "selection_order", "label"]
    ).reset_index(drop=True)


def apply_filters(df: pd.DataFrame, splits: list[str] | None, horizons: list[str] | None) -> pd.DataFrame:
    filtered = df.copy()
    if splits:
        filtered = filtered[filtered["split"].isin(splits)]
    if horizons:
        filtered = filtered[filtered["horizon"].isin(horizons)]
    return filtered.reset_index(drop=True)


def save_summary(df: pd.DataFrame, output_dir: Path, metrics: list[str]):
    keep_cols = [
        "horizon",
        "split",
        "group",
        "run_name",
        "selection",
        "label",
        "metrics_path",
    ] + metrics
    summary = df[keep_cols].copy()
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)


def sort_for_display(sub_df: pd.DataFrame) -> pd.DataFrame:
    return sub_df.sort_values(
        ["group_order", "run_name", "selection_order", "label"]
    ).reset_index(drop=True)


def plot_metric_panels(df: pd.DataFrame, metric: str, output_dir: Path):
    metric_df = df.dropna(subset=[metric]).copy()
    if metric_df.empty:
        return

    combos = list(metric_df[["horizon", "split"]].drop_duplicates().itertuples(index=False, name=None))
    n_panels = len(combos)
    fig_h = max(4.0 * n_panels, 4.5)
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, fig_h), squeeze=False)
    axes = axes.flatten()

    for ax, (horizon, split) in zip(axes, combos):
        panel = metric_df[(metric_df["horizon"] == horizon) & (metric_df["split"] == split)].copy()
        panel = sort_for_display(panel)

        values = panel[metric].to_numpy(dtype=float)
        labels = panel["label"].tolist()
        colors = (
            ["#4C78A8"] * len(values)
            if metric == "mae"
            else ["#59A14F" if value >= 0 else "#E15759" for value in values]
        )

        y_pos = np.arange(len(panel))
        ax.barh(y_pos, values, color=colors, alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{METRIC_TITLES.get(metric, metric)} | {horizon.upper()} | {split}")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        if metric != "mae":
            ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.invert_yaxis()

        span = np.nanmax(np.abs(values)) if len(values) else 0.0
        offset = max(span * 0.02, 0.0005)
        for idx, value in enumerate(values):
            text_x = value + offset if value >= 0 else value - offset
            ha = "left" if value >= 0 else "right"
            ax.text(text_x, idx, f"{value:.4f}", va="center", ha=ha, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f"{metric}_comparison.png", dpi=220)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, metrics: list[str], output_dir: Path):
    heat_df = df.dropna(subset=metrics, how="all").copy()
    if heat_df.empty:
        return

    labels = heat_df["label"].tolist()
    row_suffix = [f"{h.upper()} | {s}" for h, s in zip(heat_df["horizon"], heat_df["split"])]
    row_labels = [f"{label} ({suffix})" for label, suffix in zip(labels, row_suffix)]
    values = heat_df[metrics].to_numpy(dtype=float)

    fig_h = max(0.5 * len(row_labels) + 2.5, 4.5)
    fig_w = max(2.4 * len(metrics) + 3.5, 8.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRIC_TITLES.get(metric, metric) for metric in metrics], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Evaluation Metrics Heatmap")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = "nan" if math.isnan(value) else f"{value:.4f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, shrink=0.9)
    plt.tight_layout()
    fig.savefig(output_dir / "metrics_heatmap.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(root)
    df = apply_filters(df, args.splits, args.horizons)
    if df.empty:
        raise RuntimeError("No metrics remain after applying filters.")

    save_summary(df, output_dir, args.metrics)
    for metric in args.metrics:
        plot_metric_panels(df, metric, output_dir)
    plot_heatmap(df, args.metrics, output_dir)

    print(f"Loaded {len(df)} metrics rows from {root}")
    print(f"Saved summary to {output_dir / 'metrics_summary.csv'}")
    for metric in args.metrics:
        print(f"Saved figure: {output_dir / f'{metric}_comparison.png'}")
    print(f"Saved figure: {output_dir / 'metrics_heatmap.png'}")


if __name__ == "__main__":
    main()
