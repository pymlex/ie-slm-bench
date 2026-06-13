from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ie_slm_bench.metrics import safe_model_filename


METRIC_COLUMNS = [
    "strict_exact_match",
    "field_f1",
    "null_field_accuracy",
    "hallucination_rate",
    "schema_validity_rate",
    "entity_f1",
]

METRIC_LABELS = {
    "strict_exact_match": "Strict EM",
    "field_f1": "Field F1",
    "null_field_accuracy": "Null-field acc",
    "hallucination_rate": "Hallucination",
    "schema_validity_rate": "Schema valid",
    "entity_f1": "Entity F1",
}


def load_summary_frames(run_dir: Path) -> pd.DataFrame:
    frames = []
    for benchmark in ("nerel", "runne"):
        benchmark_dir = run_dir / benchmark
        if not benchmark_dir.exists():
            continue
        for path in benchmark_dir.glob("metrics_summary_*.csv"):
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _plot_metric_groups(
    ax,
    subset: pd.DataFrame,
    metric_names: list[str],
    title: str,
) -> None:
    models = subset["model_id"].tolist()
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(models), 1)
    for model_index, model_id in enumerate(models):
        row = subset[subset["model_id"] == model_id].iloc[0]
        values = [row[metric] for metric in metric_names]
        offsets = x - 0.4 + width / 2 + model_index * width
        ax.bar(
            offsets,
            values,
            width=width,
            label=model_id,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[name] for name in metric_names], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))


def plot_benchmark_metrics(summary: pd.DataFrame, benchmark: str, out_path: Path) -> None:
    subset = summary[summary["benchmark"] == benchmark].copy()
    if subset.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    _plot_metric_groups(
        axes[0],
        subset,
        METRIC_COLUMNS[:4],
        title=f"{benchmark.upper()} metrics (1/2)",
    )
    _plot_metric_groups(
        axes[1],
        subset,
        METRIC_COLUMNS[4:],
        title=f"{benchmark.upper()} metrics (2/2)",
    )
    fig.subplots_adjust(right=0.72, hspace=0.35)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_field_f1_by_label(per_label: pd.DataFrame, benchmark: str, out_path: Path) -> None:
    subset = per_label[per_label["benchmark"] == benchmark]
    if subset.empty:
        return
    pivot = subset.groupby(["model_id", "label"])["f1"].mean().reset_index()
    labels = sorted(pivot["label"].unique())
    models = sorted(pivot["model_id"].unique())
    x = np.arange(len(labels))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(14, 5))
    for model_index, model_id in enumerate(models):
        model_rows = pivot[pivot["model_id"] == model_id]
        values = [model_rows[model_rows["label"] == label]["f1"].mean() if label in model_rows["label"].values else 0.0 for label in labels]
        offsets = x - 0.4 + width / 2 + model_index * width
        ax.bar(offsets, values, width=width, label=model_id)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(f"{benchmark.upper()} field F1 by label")
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.subplots_adjust(right=0.72)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(run_dir: Path, assets_dir: Path) -> pd.DataFrame:
    assets_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary_frames(run_dir)
    if summary.empty:
        return summary
    summary.to_csv(assets_dir / "summary.csv", index=False)
    for benchmark in ("nerel", "runne"):
        plot_benchmark_metrics(
            summary,
            benchmark=benchmark,
            out_path=assets_dir / f"{benchmark}_metrics.png",
        )
    per_label_frames = []
    for benchmark in ("nerel", "runne"):
        benchmark_dir = run_dir / benchmark
        for path in benchmark_dir.glob("metrics_label_*.csv"):
            per_label_frames.append(pd.read_csv(path))
    if per_label_frames:
        per_label = pd.concat(per_label_frames, ignore_index=True)
        for benchmark in ("nerel", "runne"):
            plot_field_f1_by_label(
                per_label,
                benchmark=benchmark,
                out_path=assets_dir / f"{benchmark}_field_f1_by_label.png",
            )
    return summary
