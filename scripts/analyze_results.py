from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ie_slm_bench.config import RUN_DIR
from ie_slm_bench.plots import DISPLAY_NAMES, FIELD_GROUPS, generate_all_plots


MODEL_FILES = {
    "Qwen/Qwen3-1.7B": "Qwen__Qwen3-1.7B",
    "numind/NuExtract-2.0-2B": "numind__NuExtract-2.0-2B",
    "LiquidAI/LFM2-1.2B-Extract": "LiquidAI__LFM2-1.2B-Extract",
}


def build_analysis(run_dir: Path) -> dict:
    per_model: dict = {}
    groups: dict = {}
    for model_id, file_suffix in MODEL_FILES.items():
        label_path = run_dir / f"metrics_label_{file_suffix}.csv"
        example_path = run_dir / f"metrics_example_{file_suffix}.csv"
        label_frame = pd.read_csv(label_path)
        example_frame = pd.read_csv(example_path)
        label_macro = label_frame.groupby("label")["f1"].mean()
        per_model[model_id] = {
            "macro_field_f1": float(label_macro.mean()),
            "top_fields": label_macro.sort_values(ascending=False).head(8).to_dict(),
            "zero_f1_fields": [label for label, score in label_macro.items() if score == 0.0],
            "zero_f1_count": int((label_macro == 0.0).sum()),
            "label_count": int(len(label_macro)),
            "sem_docs": int(example_frame["strict_exact_match"].sum()),
            "mean_entity_f1": float(example_frame["entity_f1"].mean()),
            "mean_null_acc": float(example_frame["null_field_accuracy"].mean()),
            "mean_hallucination": float(example_frame["hallucination_rate"].mean()),
            "schema_valid_rate": float(example_frame["schema_valid"].mean()),
        }
        group_scores: dict[str, float] = {}
        for group_name, fields in FIELD_GROUPS.items():
            field_scores = [label_macro.get(field, 0.0) for field in fields if field in label_macro.index]
            group_scores[group_name] = float(sum(field_scores) / len(field_scores)) if field_scores else 0.0
        groups[model_id] = group_scores
    return {"per_model": per_model, "groups": groups}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets_dir = args.results_root / "assets"
    summary = generate_all_plots(args.run_dir, assets_dir)
    analysis = build_analysis(args.run_dir)
    (args.results_root / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"Wrote {args.results_root / 'analysis.json'}")


if __name__ == "__main__":
    main()
