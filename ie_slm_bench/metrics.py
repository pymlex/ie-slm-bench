from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from ie_slm_bench.config import BENCHMARK
from ie_slm_bench.normalize import normalize_field_value
from ie_slm_bench.parsers import extract_json_object
from schemas.bank_client import BankClientExtraction


def safe_model_filename(model_id: str) -> str:
    return model_id.replace("/", "__")


def normalize_value(path: str, value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return normalize_field_value(path, stripped)
    return normalize_field_value(path, str(value))


def flatten_gold(model: BankClientExtraction) -> dict[str, str | None]:
    raw = model.model_dump(by_alias=True, exclude_none=False)
    flat: dict[str, str | None] = {}

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                walk(next_prefix, nested)
            return
        flat[prefix] = normalize_value(prefix, value)

    walk("", raw)
    flat.pop("", None)
    return flat


def parse_prediction(
    raw_text: str,
    model_cls: type[BaseModel],
) -> tuple[BaseModel | None, bool]:
    stripped = raw_text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return model_cls.model_validate_json(stripped), True
        except ValidationError:
            pass
    return try_validate_output(raw_text, model_cls)


def try_validate_output(
    raw_text: str,
    model_cls: type[BaseModel],
) -> tuple[BaseModel | None, bool]:
    payload = extract_json_object(raw_text)
    try:
        parsed = model_cls.model_validate(payload)
    except ValidationError:
        return None, False
    return parsed, True


def strict_exact_match(gold: BankClientExtraction, pred: BankClientExtraction) -> bool:
    gold_flat = flatten_gold(gold)
    pred_flat = flatten_gold(pred)
    keys = sorted(set(gold_flat) | set(pred_flat))
    return all(gold_flat.get(key) == pred_flat.get(key) for key in keys)


def field_sets(gold: BankClientExtraction, pred: BankClientExtraction) -> tuple[dict[str, set], dict[str, set]]:
    gold_flat = flatten_gold(gold)
    pred_flat = flatten_gold(pred)
    gold_by_label: dict[str, set] = defaultdict(set)
    pred_by_label: dict[str, set] = defaultdict(set)
    for key, value in gold_flat.items():
        if value is not None:
            gold_by_label[key].add(value)
    for key, value in pred_flat.items():
        if value is not None:
            pred_by_label[key].add(value)
    return gold_by_label, pred_by_label


def prf1(gold_set: set, pred_set: set) -> tuple[float, float, float]:
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def null_field_metrics(gold: BankClientExtraction, pred: BankClientExtraction) -> tuple[float, float]:
    gold_flat = flatten_gold(gold)
    pred_flat = flatten_gold(pred)
    keys = sorted(set(gold_flat) | set(pred_flat))
    gold_null_flags = np.array([gold_flat.get(key) is None for key in keys], dtype=bool)
    pred_null_flags = np.array([pred_flat.get(key) is None for key in keys], dtype=bool)
    null_accuracy = float(np.mean(gold_null_flags == pred_null_flags))
    hallucination_mask = gold_null_flags & ~pred_null_flags
    hallucination_rate = float(np.mean(hallucination_mask))
    return null_accuracy, hallucination_rate


def entity_level_sets(gold: BankClientExtraction, pred: BankClientExtraction) -> tuple[set, set]:
    gold_flat = flatten_gold(gold)
    pred_flat = flatten_gold(pred)
    gold_set = {(key, value) for key, value in gold_flat.items() if value is not None}
    pred_set = {(key, value) for key, value in pred_flat.items() if value is not None}
    return gold_set, pred_set


def per_label_prf1(gold: BankClientExtraction, pred: BankClientExtraction) -> pd.DataFrame:
    gold_by_label, pred_by_label = field_sets(gold, pred)
    labels = sorted(set(gold_by_label) | set(pred_by_label))
    rows = []
    for label in labels:
        precision, recall, f1 = prf1(
            gold_by_label.get(label, set()),
            pred_by_label.get(label, set()),
        )
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(
    frame: pd.DataFrame,
    model_cls: type[BaseModel],
    model_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_example_rows = []
    per_label_rows = []
    for _, row in frame.iterrows():
        gold = BankClientExtraction.model_validate_json(row["gold_json"])
        pred, schema_valid = parse_prediction(row["pred_raw"], model_cls)
        if pred is None:
            pred = BankClientExtraction()
        gold_set, pred_set = entity_level_sets(gold, pred)
        entity_precision, entity_recall, entity_f1 = prf1(gold_set, pred_set)
        null_accuracy, hallucination_rate = null_field_metrics(gold, pred)
        per_example_rows.append(
            {
                "benchmark": BENCHMARK,
                "model_id": model_id,
                "doc_id": row["doc_id"],
                "strict_exact_match": float(strict_exact_match(gold, pred)),
                "entity_precision": entity_precision,
                "entity_recall": entity_recall,
                "entity_f1": entity_f1,
                "null_field_accuracy": null_accuracy,
                "hallucination_rate": hallucination_rate,
                "schema_valid": float(schema_valid),
            }
        )
        label_metrics = per_label_prf1(gold, pred)
        label_metrics["benchmark"] = BENCHMARK
        label_metrics["model_id"] = model_id
        label_metrics["doc_id"] = row["doc_id"]
        per_label_rows.append(label_metrics)
    per_example = pd.DataFrame(per_example_rows)
    per_label = pd.concat(per_label_rows, ignore_index=True)
    return per_example, per_label


def aggregate_metrics(per_example: pd.DataFrame, per_label: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for model_id in per_example["model_id"].unique():
        subset = per_example[per_example["model_id"] == model_id]
        label_subset = per_label[per_label["model_id"] == model_id]
        label_macro = label_subset.groupby("label")[["precision", "recall", "f1"]].mean().mean()
        summary_rows.append(
            {
                "benchmark": BENCHMARK,
                "model_id": model_id,
                "strict_exact_match": subset["strict_exact_match"].mean(),
                "field_precision": label_macro["precision"],
                "field_recall": label_macro["recall"],
                "field_f1": label_macro["f1"],
                "null_field_accuracy": subset["null_field_accuracy"].mean(),
                "hallucination_rate": subset["hallucination_rate"].mean(),
                "schema_validity_rate": subset["schema_valid"].mean(),
                "entity_f1": subset["entity_f1"].mean(),
            }
        )
    return pd.DataFrame(summary_rows)
