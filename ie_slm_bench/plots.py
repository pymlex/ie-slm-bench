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

ADDRESS_LABEL_PREFIXES = (
    "Адрес регистрации.",
    "Адрес фактического проживания.",
)

ADDRESS_PANEL_TITLES = {
    "Адрес регистрации.": "Адрес регистрации",
    "Адрес фактического проживания.": "Адрес фактического проживания",
}

MAX_LABELS_PER_PANEL = 8


def _split_fields(
    title: str,
    fields: list[str],
    max_width: int = MAX_LABELS_PER_PANEL,
) -> list[tuple[str, list[str]]]:
    if not fields:
        return []
    if len(fields) <= max_width:
        return [(title, fields)]
    total_parts = (len(fields) + max_width - 1) // max_width
    panels: list[tuple[str, list[str]]] = []
    for part_index, start in enumerate(range(0, len(fields), max_width)):
        chunk = fields[start:start + max_width]
        part_title = f"{title} ({part_index + 1}/{total_parts})"
        panels.append((part_title, chunk))
    return panels


def _build_label_plot_panels(
    all_labels: list[str],
    max_width: int = MAX_LABELS_PER_PANEL,
) -> list[tuple[str, list[str]]]:
    panels: list[tuple[str, list[str]]] = []
    covered: set[str] = set()
    label_set = set(all_labels)
    for group_name, fields in FIELD_GROUPS.items():
        field_labels = [field for field in fields if field in label_set]
        covered.update(field_labels)
        panels.extend(_split_fields(group_name, field_labels, max_width))
    for prefix in ADDRESS_LABEL_PREFIXES:
        address_fields = sorted(label for label in all_labels if label.startswith(prefix))
        covered.update(address_fields)
        if address_fields:
            title = ADDRESS_PANEL_TITLES.get(prefix, prefix.rstrip("."))
            panels.extend(_split_fields(title, address_fields, max_width))
    for label in sorted(label_set - covered):
        panels.extend(_split_fields(label, [label], max_width))
    return panels


def _legend_below(fig, axes, ncol: int = 3, fontsize: float = 8) -> None:
    axis_list = list(axes) if hasattr(axes, "__iter__") and not isinstance(axes, plt.Axes) else [axes]
    handles, labels = axis_list[0].get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncol,
        fontsize=fontsize,
    )
    fig.subplots_adjust(bottom=0.1)


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
    fig.subplots_adjust(hspace=0.35)
    _legend_below(fig, axes, ncol=3, fontsize=7)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_label_panel(
    ax,
    pivot: pd.DataFrame,
    models: list[str],
    field_labels: list[str],
    title: str,
) -> None:
    if not field_labels:
        ax.set_axis_off()
        return
    x = np.arange(len(field_labels))
    width = 0.8 / max(len(models), 1)
    for model_index, model_id in enumerate(models):
        model_rows = pivot[pivot["model_id"] == model_id]
        values = [
            float(model_rows[model_rows["label"] == label]["f1"].mean())
            if label in model_rows["label"].values
            else 0.0
            for label in field_labels
        ]
        offsets = x - 0.4 + width / 2 + model_index * width
        display_name = DISPLAY_NAMES.get(model_id, model_id)
        ax.bar(offsets, values, width=width, label=display_name)
    ax.set_xticks(x)
    ax.set_xticklabels(field_labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.5)


def plot_field_f1_by_label(per_label: pd.DataFrame, out_path: Path) -> None:
    if per_label.empty:
        return
    pivot = per_label.groupby(["model_id", "label"])["f1"].mean().reset_index()
    all_labels = sorted(pivot["label"].unique())
    models = sorted(pivot["model_id"].unique())
    panels = _build_label_plot_panels(all_labels)
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.2 * len(panels)))
    axes = np.atleast_1d(axes)
    for axis, (panel_title, field_labels) in zip(axes, panels):
        _plot_label_panel(
            axis,
            pivot,
            models,
            field_labels,
            f"{BENCHMARK} field F1 — {panel_title}",
        )
    fig.subplots_adjust(hspace=0.55)
    _legend_below(fig, axes, ncol=3, fontsize=8)
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
    ax.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    fig.subplots_adjust(bottom=0.22)
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
