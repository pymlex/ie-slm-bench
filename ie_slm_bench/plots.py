from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ie_slm_bench.config import BENCHMARK


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

FIELD_GROUPS = {
    "Identity": [
        "Фамилия",
        "Имя",
        "Отчество",
        "Пол",
        "Дата рождения",
        "Год рождения",
        "Место рождения",
        "Гражданство",
    ],
    "Passport": [
        "Серия и номер паспорта",
        "Кем выдан паспорт",
        "Дата выдачи паспорта",
        "Код подразделения",
    ],
    "IDs & contact": [
        "ИНН",
        "СНИЛС",
        "Номер мобильного телефона",
        "Адрес электронной почты",
    ],
    "Work": [
        "Место работы",
        "Должность на работе",
        "Стаж работы.лет",
        "Стаж работы.месяцев",
        "Ежемесячный доход",
    ],
    "Assets & family": [
        "Наличие недвижимости",
        "Наличие автомобиля",
        "Наличие кредитов/займов",
        "Количество иждивенцев",
        "Семейное положение",
    ],
}

DISPLAY_NAMES = {
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    "numind/NuExtract-2.0-2B": "NuExtract-2.0-2B",
    "LiquidAI/LFM2-1.2B-Extract": "LFM2-1.2B-Extract",
}


def load_summary_frames(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in run_dir.glob("metrics_summary_*.csv"):
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
        display_name = DISPLAY_NAMES.get(model_id, model_id)
        ax.bar(
            offsets,
            values,
            width=width,
            label=display_name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[name] for name in metric_names], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))


def plot_metrics(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    _plot_metric_groups(
        axes[0],
        summary,
        METRIC_COLUMNS[:4],
        title=f"{BENCHMARK} metrics (1/2)",
    )
    _plot_metric_groups(
        axes[1],
        summary,
        METRIC_COLUMNS[4:],
        title=f"{BENCHMARK} metrics (2/2)",
    )
    fig.subplots_adjust(right=0.72, hspace=0.35)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_field_f1_by_label(per_label: pd.DataFrame, out_path: Path) -> None:
    if per_label.empty:
        return
    pivot = per_label.groupby(["model_id", "label"])["f1"].mean().reset_index()
    labels = sorted(pivot["label"].unique())
    models = sorted(pivot["model_id"].unique())
    x = np.arange(len(labels))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(14, 5))
    for model_index, model_id in enumerate(models):
        model_rows = pivot[pivot["model_id"] == model_id]
        values = [
            model_rows[model_rows["label"] == label]["f1"].mean()
            if label in model_rows["label"].values
            else 0.0
            for label in labels
        ]
        offsets = x - 0.4 + width / 2 + model_index * width
        display_name = DISPLAY_NAMES.get(model_id, model_id)
        ax.bar(offsets, values, width=width, label=display_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(f"{BENCHMARK} field F1 by label")
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.subplots_adjust(right=0.72)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_field_group_f1(per_label: pd.DataFrame, out_path: Path) -> None:
    if per_label.empty:
        return
    label_macro = per_label.groupby(["model_id", "label"])["f1"].mean().reset_index()
    group_names = list(FIELD_GROUPS)
    models = sorted(label_macro["model_id"].unique())
    x = np.arange(len(group_names))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_index, model_id in enumerate(models):
        model_rows = label_macro[label_macro["model_id"] == model_id]
        label_scores = model_rows.set_index("label")["f1"]
        values = []
        for group_name in group_names:
            fields = FIELD_GROUPS[group_name]
            field_scores = [label_scores.get(field, 0.0) for field in fields if field in label_scores.index]
            values.append(float(np.mean(field_scores)) if field_scores else 0.0)
        offsets = x - 0.4 + width / 2 + model_index * width
        display_name = DISPLAY_NAMES.get(model_id, model_id)
        ax.bar(offsets, values, width=width, label=display_name)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{BENCHMARK} macro field F1 by field group")
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.subplots_adjust(right=0.75)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(run_dir: Path, assets_dir: Path) -> pd.DataFrame:
    assets_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary_frames(run_dir)
    if summary.empty:
        return summary
    summary.to_csv(assets_dir / "summary.csv", index=False)
    plot_metrics(summary, assets_dir / "ru_bank_ie_metrics.png")
    per_label_frames = []
    for path in run_dir.glob("metrics_label_*.csv"):
        per_label_frames.append(pd.read_csv(path))
    if per_label_frames:
        per_label = pd.concat(per_label_frames, ignore_index=True)
        plot_field_f1_by_label(per_label, assets_dir / "ru_bank_ie_field_f1_by_label.png")
        plot_field_group_f1(per_label, assets_dir / "ru_bank_ie_field_group_f1.png")
    return summary
