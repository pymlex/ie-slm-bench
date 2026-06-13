from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class CoverageValidation(BaseModel):
    all_present: bool
    missing_fields: list[str]
    justification: str


def build_justification(all_present: bool, missing_fields: list[str]) -> str:
    if all_present:
        return "Каждое непустое поле gold присутствует в тексте клиента."
    joined = ", ".join(missing_fields)
    return f"В тексте клиента не найдены поля gold: {joined}."


def parse_missing_fields(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            return [str(item) for item in parsed]
        return [stripped]
    return [str(raw)]


def coverage_from_row(row: dict) -> CoverageValidation:
    if "validation_json" in row:
        return CoverageValidation.model_validate_json(row["validation_json"])
    all_present = bool(row.get("coverage_ok", row.get("all_present", False)))
    missing_fields = parse_missing_fields(row.get("missing_fields"))
    justification = row.get("justification")
    if justification is None:
        justification = build_justification(all_present, missing_fields)
    return CoverageValidation(
        all_present=all_present,
        missing_fields=missing_fields,
        justification=str(justification),
    )


def generation_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "text": row["text"],
        "gold_json": row["gold_json"],
    }


def validated_row(row: dict, coverage: CoverageValidation) -> dict:
    generation = generation_row(row)
    generation["validation_json"] = coverage.model_dump_json()
    return generation


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize_dataset(data_dir: Path) -> tuple[int, int]:
    stage3_path = data_dir / "stage3_validated.jsonl"
    if not stage3_path.exists():
        raise FileNotFoundError(f"Missing {stage3_path}")

    source_rows = load_jsonl(stage3_path)
    stage3_rows = []
    test_rows = []

    for row in source_rows:
        coverage = coverage_from_row(row)
        validated = validated_row(row, coverage)
        stage3_rows.append(validated)
        if coverage.all_present:
            test_rows.append(validated)

    save_jsonl(stage3_rows, stage3_path)
    save_jsonl(test_rows, data_dir / "test.jsonl")
    pd.DataFrame(test_rows).to_csv(data_dir / "test.csv", index=False)
    return len(stage3_rows), len(test_rows)
