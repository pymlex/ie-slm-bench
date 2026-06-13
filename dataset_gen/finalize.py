from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from dataset_gen.text_split import cyr_ratio, split_reasoning_and_text
from schemas.bank_client import BankClientExtraction


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


def combined_generation_text(row: dict) -> str:
    if row.get("combined_text"):
        return str(row["combined_text"])
    reasoning = row.get("reasoning")
    text = row.get("text", "")
    if reasoning:
        return f"{reasoning}\n\n{text}".strip()
    return str(text)


def normalize_generation_row(row: dict) -> dict:
    if (
        row.get("reasoning") is not None
        and row.get("text") is not None
        and not str(row.get("text", "")).strip().lower().startswith("the user wants")
    ):
        return {
            "id": row["id"],
            "reasoning": str(row["reasoning"]),
            "text": str(row["text"]),
            "gold_json": row["gold_json"],
        }

    combined = combined_generation_text(row)
    if (
        not combined.strip().lower().startswith("the user")
        and cyr_ratio(combined) > 0.4
        and "reasoning" not in row
    ):
        return {
            "id": row["id"],
            "reasoning": "",
            "text": combined.strip(),
            "gold_json": row["gold_json"],
        }

    reasoning, text = split_reasoning_and_text(combined)
    return {
        "id": row["id"],
        "reasoning": reasoning,
        "text": text,
        "gold_json": row["gold_json"],
    }


def generation_row(row: dict) -> dict:
    normalized = normalize_generation_row(row)
    return {
        "id": normalized["id"],
        "reasoning": normalized["reasoning"],
        "text": normalized["text"],
        "gold_json": normalized["gold_json"],
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


def recheck_coverage_batch(
    rows: list[dict],
    batch_size: int,
) -> list[CoverageValidation]:
    from dataset_gen.llm import GeneratorBackend

    backend = GeneratorBackend(batch_size=batch_size)
    backend.load()
    coverages: list[CoverageValidation] = []
    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        golds = [BankClientExtraction.model_validate_json(row["gold_json"]) for row in batch]
        texts = [row["text"] for row in batch]
        checks = backend.check_coverage_batch(texts, golds)
        coverages.extend(checks)
    backend.unload()
    return coverages


def finalize_dataset(
    data_dir: Path,
    recheck_coverage: bool = False,
    batch_size: int = 32,
) -> tuple[int, int]:
    stage2_path = data_dir / "stage2_pairs.jsonl"
    stage3_path = data_dir / "stage3_validated.jsonl"
    if not stage3_path.exists():
        raise FileNotFoundError(f"Missing {stage3_path}")

    if stage2_path.exists():
        stage2_rows = [normalize_generation_row(row) for row in load_jsonl(stage2_path)]
        save_jsonl(stage2_rows, stage2_path)

    source_rows = load_jsonl(stage3_path)
    normalized_rows = [normalize_generation_row(row) for row in source_rows]

    if recheck_coverage:
        coverages = recheck_coverage_batch(normalized_rows, batch_size=batch_size)
    else:
        coverages = [coverage_from_row(row) for row in source_rows]

    stage3_rows = []
    test_rows = []
    for row, coverage in zip(normalized_rows, coverages):
        validated = validated_row(row, coverage)
        stage3_rows.append(validated)
        if coverage.all_present and validated["text"].strip():
            test_rows.append(validated)

    save_jsonl(stage3_rows, stage3_path)
    save_jsonl(test_rows, data_dir / "test.jsonl")
    pd.DataFrame(test_rows).to_csv(data_dir / "test.csv", index=False)
    return len(stage3_rows), len(test_rows)

