#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


METRICS = [
    ("mean_rank_ic", "rank_ic"),
    ("mean_ic", "ic"),
    ("mae", "mae"),
]
FAMILY_ORDER = {"kronos": 0, "timesfm": 1}
CONTEXT_ORDER = {20: 0, 32: 1}
COLOR_MAP = {"kronos": "#4C78A8", "timesfm": "#F28E2B"}
VALID_SPLITS = {"val", "test", "all"}
AUTO_SPLIT_PRIORITY = {"all": 0, "test": 1, "val": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Kronos and TimesFM stage-1 metrics with fixed experiment ordering."
    )
    parser.add_argument("kronos_root", type=str, help="Root path for Kronos stage-1 outputs.")
    parser.add_argument("timesfm_root", type=str, help="Root path for TimesFM stage-1 outputs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="zlab/results/timesfm_kronos_stage1/comparisons",
        help="Directory for comparison csv and plots.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "val", "test", "all"],
        help="Split selection mode. auto means: prefer all, then test, then val.",
    )
    return parser.parse_args()


def _normalize_metrics_payload(payload: dict) -> dict:
    summary = payload.get("summary")
    metric_source = summary if isinstance(summary, dict) else payload

    normalized = dict(payload)
    normalized["mean_rank_ic"] = metric_source.get("mean_rank_ic")
    normalized["mean_ic"] = metric_source.get("mean_ic")
    normalized["mae"] = metric_source.get("mae", metric_source.get("mae_return"))
    normalized["metric_source"] = "summary" if isinstance(summary, dict) else "root"
    return normalized


def _scan_one_family(root: Path, family: str, split: str) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        rel = metrics_path.relative_to(root)
        parts = rel.parts
        if len(parts) < 4 or parts[-2] not in VALID_SPLITS:
            continue
        if split != "auto" and parts[-2] != split:
            continue

        ctx_match = re.search(r"evaluations_ctx(\d+)", parts[0])
        horizon_match = re.fullmatch(r"h(\d+)", parts[1])
        if ctx_match is None or horizon_match is None:
            continue

        context = int(ctx_match.group(1))
        horizon = int(horizon_match.group(1))
        if context not in CONTEXT_ORDER:
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload = _normalize_metrics_payload(payload)

        run_name = parts[2]
        label = f"{family}_ctx{context}"
        rows.append(
            {
                "family": family,
                "context": context,
                "horizon": horizon,
                "split": parts[-2],
                "run_name": run_name,
                "label": label,
                "metrics_path": str(metrics_path),
                "display_order": FAMILY_ORDER[family] * 10 + CONTEXT_ORDER[context],
                **payload,
            }
        )
    return rows


def load_metrics(kronos_root: Path, timesfm_root: Path, split: str) -> pd.DataFrame:
    rows = []
    rows.extend(_scan_one_family(kronos_root, "kronos", split))
    rows.extend(_scan_one_family(timesfm_root, "timesfm", split))
    if not rows:
        raise FileNotFoundError("No matching metrics.json found under the provided roots.")

    df = pd.DataFrame(rows)
    if split == "auto":
        df["split_priority"] = df["split"].map(AUTO_SPLIT_PRIORITY).fillna(99)
        df = (
            df.sort_values(["horizon", "display_order", "split_priority", "run_name"])
            .drop_duplicates(subset=["family", "context", "horizon"], keep="first")
            .reset_index(drop=True)
        )
    else:
        df = (
            df.sort_values(["horizon", "display_order", "run_name"])
            .drop_duplicates(subset=["family", "context", "horizon", "split"], keep="first")
            .reset_index(drop=True)
        )
    return df


def plot_metric(df: pd.DataFrame, horizon: int, metric_key: str, metric_title: str, output_path: Path):
    panel = df[df["horizon"] == horizon].copy()
    panel = panel.sort_values("display_order").reset_index(drop=True)
    if panel.empty:
        return

    values = panel[metric_key].to_numpy(dtype=float)
    labels = panel["label"].tolist()
    colors = [COLOR_MAP[row.family] for row in panel.itertuples()]

    fig_h = max(2.4, 0.5 * len(panel) + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, fig_h), constrained_layout=True)

    y = np.arange(len(panel)) * 0.85
    ax.barh(y, values, height=0.48, color=colors, alpha=0.92)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(f"{metric_title} | h{horizon}", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=6))
    ax.margins(y=0.03)

    if metric_key != "mae":
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)

    min_value = float(np.nanmin(values)) if len(values) else 0.0
    max_value = float(np.nanmax(values)) if len(values) else 0.0
    left_anchor = min(0.0, min_value) if metric_key != "mae" else max(0.0, min_value)
    right_anchor = max(0.0, max_value)
    value_span = max(right_anchor - left_anchor, max(abs(min_value), abs(max_value)), 0.001)
    x_pad = max(value_span * 0.22, 0.0015)
    x_left = left_anchor - (x_pad if metric_key != "mae" else x_pad * 0.15)
    x_right = right_anchor + x_pad
    ax.set_xlim(x_left, x_right)

    offset = max(value_span * 0.03, 0.0005)
    for idx, value in enumerate(values):
        text_x = value + offset if value >= 0 else value - offset
        ha = "left" if value >= 0 else "right"
        text_y = y[idx]
        ax.text(text_x, text_y, f"{value:.4f}", va="center", ha=ha, fontsize=9)

    legend_handles = [
        Patch(facecolor=COLOR_MAP["kronos"], label="kronos"),
        Patch(facecolor=COLOR_MAP["timesfm"], label="timesfm"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=True,
        fontsize=9,
        handlelength=1.4,
        columnspacing=1.6,
    )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()

    kronos_root = Path(args.kronos_root).resolve()
    timesfm_root = Path(args.timesfm_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(kronos_root, timesfm_root, args.split)
    df.to_csv(output_dir / f"metrics_summary_{args.split}.csv", index=False)

    for horizon in (1, 5):
        horizon_df = df[df["horizon"] == horizon]
        if horizon_df.empty:
            continue
        for metric_key, metric_title in METRICS:
            plot_metric(
                df,
                horizon,
                metric_key,
                metric_title,
                output_dir / f"{metric_title}_h{horizon}_{args.split}.png",
            )

    print(f"Loaded {len(df)} experiment rows")
    print(f"Saved summary: {output_dir / f'metrics_summary_{args.split}.csv'}")
    for horizon in (1, 5):
        for _, metric_title in METRICS:
            plot_path = output_dir / f"{metric_title}_h{horizon}_{args.split}.png"
            if plot_path.exists():
                print(f"Saved figure: {plot_path}")


if __name__ == "__main__":
    main()
