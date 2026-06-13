from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from ie_slm_bench.config import BENCHMARK
from ie_slm_bench.parsers import extract_json_object
from schemas.runne import RunneExtraction


def safe_model_filename(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_extraction(raw_json: str, model_cls: type[BaseModel]) -> BaseModel:
    payload = json.loads(raw_json)
    return model_cls.model_validate(payload)


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


def entity_key(entity) -> tuple:
    return (entity.start, entity.end, entity.type)


def strict_exact_match(gold: BaseModel, pred: BaseModel) -> bool:
    gold_obj = RunneExtraction.model_validate(gold.model_dump())
    pred_obj = RunneExtraction.model_validate(pred.model_dump())
    gold_entities = sorted(entity_key(item) for item in gold_obj.entities)
    pred_entities = sorted(entity_key(item) for item in pred_obj.entities)
    return gold_entities == pred_entities


def entity_level_sets(gold: BaseModel, pred: BaseModel) -> tuple[set, set]:
    gold_set = {entity_key(item) for item in gold.entities}
    pred_set = {entity_key(item) for item in pred.entities}
    return gold_set, pred_set


def prf1(gold_set: set, pred_set: set) -> tuple[float, float, float]:
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def null_field_records(gold: BaseModel, pred: BaseModel) -> list[tuple[str, bool, bool]]:
    gold_obj = RunneExtraction.model_validate(gold.model_dump())
    pred_obj = RunneExtraction.model_validate(pred.model_dump())
    gold_empty = len(gold_obj.entities) == 0
    pred_empty = len(pred_obj.entities) == 0
    return [("entities", gold_empty, pred_empty)]


def null_field_metrics(gold: BaseModel, pred: BaseModel) -> tuple[float, float]:
    records = null_field_records(gold, pred)
    gold_null_flags = np.array([item[1] for item in records], dtype=bool)
    pred_null_flags = np.array([item[2] for item in records], dtype=bool)
    null_accuracy = float(np.mean(gold_null_flags == pred_null_flags))
    hallucination_mask = gold_null_flags & ~pred_null_flags
    hallucination_rate = float(np.mean(hallucination_mask))
    return null_accuracy, hallucination_rate


def label_value_sets(gold: BaseModel, pred: BaseModel) -> tuple[dict[str, set], dict[str, set]]:
    gold_by_label: dict[str, set] = defaultdict(set)
    pred_by_label: dict[str, set] = defaultdict(set)
    for item in gold.entities:
        gold_by_label[f"entity:{item.type}"].add(entity_key(item))
    for item in pred.entities:
        pred_by_label[f"entity:{item.type}"].add(entity_key(item))
    return gold_by_label, pred_by_label


def per_label_prf1(gold: BaseModel, pred: BaseModel) -> pd.DataFrame:
    gold_by_label, pred_by_label = label_value_sets(gold, pred)
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
        gold = load_extraction(row["gold_json"], model_cls)
        pred, schema_valid = try_validate_output(row["pred_raw"], model_cls)
        if pred is None:
            pred = model_cls()
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
