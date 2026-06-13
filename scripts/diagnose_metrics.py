from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from ie_slm_bench.config import RUN_DIR
from ie_slm_bench.metrics import (
    aggregate_metrics,
    evaluate_predictions,
    flatten_gold,
    parse_prediction,
    safe_model_filename,
)
from schemas.bank_client import BankClientExtraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--model-id", type=str, default=None)
    return parser.parse_args()


def diagnose_row(gold: BankClientExtraction, pred: BankClientExtraction) -> dict:
    gold_flat = flatten_gold(gold)
    pred_flat = flatten_gold(pred)
    keys = sorted(set(gold_flat) | set(pred_flat))
    gold_nonnull = 0
    matches = 0
    for key in keys:
        gold_value = gold_flat.get(key)
        pred_value = pred_flat.get(key)
        if gold_value is None:
            continue
        gold_nonnull += 1
        if gold_value == pred_value:
            matches += 1
    recall = matches / gold_nonnull if gold_nonnull else 0.0
    return {
        "gold_nonnull": gold_nonnull,
        "value_matches": matches,
        "gold_value_recall": recall,
    }


def main() -> None:
    args = parse_args()
    pred_files = sorted(args.run_dir.glob("pred_*.csv"))
    if args.model_id is not None:
        pred_files = [args.run_dir / f"pred_{safe_model_filename(args.model_id)}.csv"]

    for pred_path in pred_files:
        frame = pd.read_csv(pred_path)
        model_id = frame["model_id"].iloc[0]
        per_example, per_label = evaluate_predictions(frame, BankClientExtraction, model_id)
        summary = aggregate_metrics(per_example, per_label)
        print("=" * 80)
        print(model_id)
        print(summary.to_string(index=False))

        sample = frame.head(3)
        for _, row in sample.iterrows():
            gold = BankClientExtraction.model_validate_json(row["gold_json"])
            pred, schema_valid = parse_prediction(row["pred_raw"], BankClientExtraction)
            if pred is None:
                pred = BankClientExtraction()
            stats = diagnose_row(gold, pred)
            print(
                f"doc_id={row['doc_id']} schema_valid={schema_valid} "
                f"gold_nonnull={stats['gold_nonnull']} "
                f"matches={stats['value_matches']} "
                f"recall={stats['gold_value_recall']:.3f}"
            )
            gold_flat = flatten_gold(gold)
            pred_flat = flatten_gold(pred)
            for key in sorted(gold_flat):
                if gold_flat[key] is None:
                    continue
                print(f"  {key}: gold={gold_flat[key]} pred={pred_flat.get(key)}")


if __name__ == "__main__":
    main()
