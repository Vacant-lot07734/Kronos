#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


DEFAULT_METRICS = ["rank_ic", "ic", "rank_icir", "icir", "da", "mae", "rmse"]
METRIC_COLUMN_MAP = {
    "rank_ic": "rank_ic",
    "ic": "ic",
    "rank_icir": "rank_icir",
    "icir": "icir",
    "da": "da",
    "mae": "mae",
    "rmse": "rmse",
}
METRIC_TITLES = {
    "rank_ic": "rank_ic",
    "ic": "ic",
    "rank_icir": "rank_icir",
    "icir": "icir",
    "da": "da",
    "mae": "mae",
    "rmse": "rmse",
}
LOWER_IS_BETTER = {"mae", "rmse"}
SELECTION_ORDER = {"default": 0, "cross-entropy": 1, "rankIc": 2}
SPLIT_ORDER = {"val": 0, "test": 1}
GROUP_ORDER = {"group_a": 0, "group_b": 1, "group_c": 2}
GROUP_COLORS = {
    "group_a": "#4C78A8",
    "group_b": "#59A14F",
    "group_c": "#F28E2B",
    "other": "#9C9C9C",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics and generate fixed-order comparison plots."
    )
    parser.add_argument(
        "pred_len",
        nargs="?",
        default="1",
        choices=["1", "5"],
        help="Prediction horizon to compare. Default: 1",
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
        help="Metrics to compare. Default: rank_ic ic rank_icir icir da mae rmse",
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
        help="Optional explicit horizon filter. When omitted, pred_len controls the filter.",
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
    df = df.sort_values(
        ["horizon_num", "split_order", "group_order", "run_name", "selection_order", "label"]
    ).reset_index(drop=True)
    df["display_label"] = df.apply(
        lambda row: f"{row['label']} ({row['horizon'].upper()} | {row['split']})",
        axis=1,
    )
    df["display_order"] = np.arange(len(df))
    return df


def apply_filters(df: pd.DataFrame, splits: list[str] | None, horizons: list[str] | None) -> pd.DataFrame:
    filtered = df.copy()
    if splits:
        filtered = filtered[filtered["split"].isin(splits)]
    if horizons:
        filtered = filtered[filtered["horizon"].isin(horizons)]
    filtered = filtered.sort_values("display_order").reset_index(drop=True)
    return filtered


def validate_metrics(metrics: list[str]):
    unknown = [metric for metric in metrics if metric not in METRIC_COLUMN_MAP]
    if unknown:
        raise ValueError(f"Unsupported metrics: {unknown}")


def save_summary(df: pd.DataFrame, output_dir: Path, metrics: list[str], filename: str):
    keep_cols = [
        "horizon",
        "split",
        "group",
        "run_name",
        "selection",
        "label",
        "display_label",
        "metrics_path",
    ] + metrics
    summary = df[keep_cols].copy()
    summary.to_csv(output_dir / filename, index=False)


def select_excellent_bc(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    bc_df = df[df["group"].isin(["group_b", "group_c"])].copy()
    selected_labels: set[str] = set()

    for metric in metrics:
        metric_df = bc_df.dropna(subset=[metric]).copy()
        if metric_df.empty:
            continue
        ascending = metric in LOWER_IS_BETTER
        metric_df = metric_df.sort_values(
            [metric, "display_order"],
            ascending=[ascending, True],
        )
        top_df = metric_df.head(2)
        selected_labels.update(top_df["display_label"].tolist())

    selected_df = df[
        (df["group"] == "group_a") | (df["display_label"].isin(selected_labels))
    ].copy()
    return selected_df.sort_values("display_order").reset_index(drop=True)


def _value_offset(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.001
    span = float(np.nanmax(np.abs(values)))
    return max(span * 0.02, 0.0005)


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path, title_prefix: str):
    metric_df = df.dropna(subset=[metric]).sort_values("display_order").reset_index(drop=True)
    if metric_df.empty:
        return

    values = metric_df[metric].to_numpy(dtype=float)
    labels = metric_df["display_label"].tolist()
    colors = [GROUP_COLORS.get(group, GROUP_COLORS["other"]) for group in metric_df["group"]]

    fig_h = max(0.35 * len(metric_df) + 2.8, 5.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    y_pos = np.arange(len(metric_df))

    ax.barh(y_pos, values, color=colors, alpha=0.92)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(f"{title_prefix} | {METRIC_TITLES[metric]}")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    if metric not in LOWER_IS_BETTER:
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.invert_yaxis()

    offset = _value_offset(values)
    for idx, value in enumerate(values):
        text_x = value + offset if value >= 0 else value - offset
        ha = "left" if value >= 0 else "right"
        ax.text(text_x, idx, f"{value:.4f}", va="center", ha=ha, fontsize=8)

    legend_handles = [
        Patch(color=GROUP_COLORS["group_a"], label="group_a"),
        Patch(color=GROUP_COLORS["group_b"], label="group_b"),
        Patch(color=GROUP_COLORS["group_c"], label="group_c"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    validate_metrics(args.metrics)

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(root)
    horizons = args.horizons if args.horizons else [f"h{args.pred_len}"]
    df = apply_filters(df, args.splits, horizons)
    if df.empty:
        raise RuntimeError("No metrics remain after applying filters.")

    selected_df = select_excellent_bc(df, args.metrics)
    suffix = args.pred_len

    save_summary(df, output_dir, args.metrics, "metrics_summary_all.csv")
    save_summary(selected_df, output_dir, args.metrics, "metrics_summary_selected.csv")

    for metric in args.metrics:
        plot_metric(
            df,
            metric,
            output_dir / f"all_{METRIC_TITLES[metric]}{suffix}.png",
            f"all experiments | pred_len={args.pred_len}",
        )
        plot_metric(
            selected_df,
            metric,
            output_dir / f"selected_{METRIC_TITLES[metric]}{suffix}.png",
            f"a + selected b/c experiments | pred_len={args.pred_len}",
        )

    print(f"Loaded {len(df)} metrics rows from {root}")
    print(f"Saved summary to {output_dir / 'metrics_summary_all.csv'}")
    print(f"Saved summary to {output_dir / 'metrics_summary_selected.csv'}")
    for metric in args.metrics:
        print(f"Saved figure: {output_dir / f'all_{METRIC_TITLES[metric]}{suffix}.png'}")
        print(f"Saved figure: {output_dir / f'selected_{METRIC_TITLES[metric]}{suffix}.png'}")


if __name__ == "__main__":
    main()
