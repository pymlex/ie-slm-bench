from __future__ import annotations

import json
from pathlib import Path


def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def load_summary_table(results_root: Path) -> str:
    metrics_path = results_root / "metrics.json"
    if not metrics_path.exists():
        return ""
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload["summary"]
    lines = [
        "| Model | SEM | Field F1 | Entity F1 | Null-field acc | Hallucination | Schema valid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    display = {
        "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
        "numind/NuExtract-2.0-2B": "NuExtract-2.0-2B",
        "LiquidAI/LFM2-1.2B-Extract": "LFM2-1.2B-Extract",
    }
    for row in rows:
        name = display.get(row["model_id"], row["model_id"])
        lines.append(
            f"| {name} | {_pct(row['strict_exact_match'])} | {_pct(row['field_f1'])} | "
            f"{_pct(row['entity_f1'])} | {_pct(row['null_field_accuracy'])} | "
            f"{_pct(row['hallucination_rate'])} | {_pct(row['schema_validity_rate'])} |"
        )
    return "\n".join(lines)


def load_analysis_bullets(results_root: Path) -> str:
    analysis_path = results_root / "analysis.json"
    if not analysis_path.exists():
        return ""
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    nuextract = analysis["per_model"]["numind/NuExtract-2.0-2B"]
    qwen = analysis["per_model"]["Qwen/Qwen3-1.7B"]
    lfm = analysis["per_model"]["LiquidAI/LFM2-1.2B-Extract"]
    groups = analysis["groups"]
    bullets = [
        f"- Evaluation split $N=368$ coverage-valid pairs from 500 stage-3 rows. Hardware: NVIDIA RTX 5090, bf16 inference.",
        f"- `numind/NuExtract-2.0-2B` leads on entity F1 ({_pct(nuextract['mean_entity_f1'])}%) and macro field F1 ({_pct(nuextract['macro_field_f1'])}%). "
        f"Three documents reach strict exact match. Schema validity is 100%.",
        f"- `Qwen/Qwen3-1.7B` reaches entity F1 {_pct(qwen['mean_entity_f1'])}% with schema validity {_pct(qwen['schema_valid_rate'])}%. "
        f"Identity fields extract better than passport blocks. Null-field accuracy is {_pct(qwen['mean_null_acc'])}%.",
        f"- `LiquidAI/LFM2-1.2B-Extract` stays below {_pct(lfm['macro_field_f1'])}% macro field F1. "
        f"Passport and address groups are near zero.",
        f"- NuExtract field-group F1: IDs & contact {_pct(groups['numind/NuExtract-2.0-2B']['IDs & contact'])}%, "
        f"work {_pct(groups['numind/NuExtract-2.0-2B']['Work'])}%, passport {_pct(groups['numind/NuExtract-2.0-2B']['Passport'])}%.",
        f"- Address sub-fields with index, country and region stay at 0% F1 for NuExtract when gold omits them from client text.",
        f"- Metrics CSV and plots in this repository under `benchmark/` and `benchmark_assets/`. "
        f"GitHub mirror: [ie-slm-bench results](https://github.com/pymlex/ie-slm-bench/tree/main/results).",
    ]
    return "\n".join(bullets)


def load_plots_markdown(asset_prefix: str = "benchmark_assets") -> str:
    plots = [
        ("ru_bank_ie_metrics.png", "ru-bank-ie aggregate metrics"),
        ("ru_bank_ie_field_group_f1.png", "ru-bank-ie field group F1"),
        ("ru_bank_ie_field_f1_by_label.png", "ru-bank-ie field F1 by label"),
    ]
    blocks = []
    for filename, alt in plots:
        blocks.append(
            f"<p align=\"center\">\n"
            f"  <img src=\"{asset_prefix}/{filename}\" alt=\"{alt}\" width=\"720\" />\n"
            f"</p>"
        )
    return "\n\n".join(blocks)


def benchmark_results_markdown(results_root: Path, asset_prefix: str = "benchmark_assets") -> str:
    table = load_summary_table(results_root)
    bullets = load_analysis_bullets(results_root)
    if not table:
        return ""
    plots = load_plots_markdown(asset_prefix)
    sections = [table, bullets]
    if plots:
        sections.append("### Plots\n\n" + plots)
    return "\n\n".join(sections)
