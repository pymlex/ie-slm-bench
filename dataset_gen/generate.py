from __future__ import annotations

import argparse
import json
from pathlib import Path

from ie_slm_bench.env import load_env

load_env()

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dataset_gen.llm import GeneratorBackend
from dataset_gen.masks import build_gold_spec
from ie_slm_bench.config import DATASET_SIZE, DATA_DIR, GEN_BATCH_SIZE, SEED
from schemas.bank_client import BankClientExtraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DATASET_SIZE)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=GEN_BATCH_SIZE)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
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


def collect_surnames(gold_rows: list[dict]) -> list[str]:
    surnames = []
    for row in gold_rows:
        gold = BankClientExtraction.model_validate_json(row["gold_json"])
        if gold.surname is not None:
            surnames.append(gold.surname)
    return surnames


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_path = out_dir / "stage1_gold.jsonl"
    pair_path = out_dir / "stage2_pairs.jsonl"
    validated_path = out_dir / "stage3_validated.jsonl"

    gold_rows = load_jsonl(gold_path)
    pair_rows = load_jsonl(pair_path)
    validated_rows = load_jsonl(validated_path)

    need_backend = (
        len(gold_rows) < args.n
        or len(pair_rows) < args.n
        or len(validated_rows) < args.n
    )
    backend = None
    if need_backend:
        backend = GeneratorBackend(batch_size=args.batch_size)
        backend.load()

    if len(gold_rows) < args.n:
        used_surnames = collect_surnames(gold_rows)
        batch_starts = range(len(gold_rows), args.n, args.batch_size)
        for batch_start in tqdm(batch_starts, desc="stage1 gold", total=len(batch_starts)):
            sample_ids = list(range(batch_start, min(batch_start + args.batch_size, args.n)))
            specs = [
                build_gold_spec(
                    rng,
                    sample_id=sample_id,
                    total=args.n,
                    used_surnames=used_surnames,
                    batch_slot=slot_index,
                )
                for slot_index, sample_id in enumerate(sample_ids)
            ]
            gold_models = backend.generate_gold_batch(specs)
            for sample_id, gold in zip(sample_ids, gold_models):
                gold_rows.append(
                    {
                        "id": sample_id,
                        "gold_json": gold.model_dump_json(by_alias=True, exclude_none=False),
                    }
                )
                if gold.surname is not None:
                    used_surnames.append(gold.surname)
            save_jsonl(gold_rows, gold_path)
        print(f"stage1 complete: {len(gold_rows)} gold rows")
    else:
        print(f"stage1 resume: {len(gold_rows)} gold rows")

    if len(pair_rows) < args.n:
        pending_gold = gold_rows[len(pair_rows) :]
        batch_starts = range(0, len(pending_gold), args.batch_size)
        for batch_start in tqdm(batch_starts, desc="stage2 texts", total=len(batch_starts)):
            batch = pending_gold[batch_start : batch_start + args.batch_size]
            golds = [BankClientExtraction.model_validate_json(row["gold_json"]) for row in batch]
            texts = backend.generate_text_batch(golds)
            for row, text in zip(batch, texts):
                pair_rows.append(
                    {
                        "id": row["id"],
                        "text": text,
                        "gold_json": row["gold_json"],
                    }
                )
            save_jsonl(pair_rows, pair_path)
        print(f"stage2 complete: {len(pair_rows)} pairs")
    else:
        print(f"stage2 resume: {len(pair_rows)} pairs")

    if len(validated_rows) < args.n:
        pending_pairs = pair_rows[len(validated_rows) :]
        batch_starts = range(0, len(pending_pairs), args.batch_size)
        for batch_start in tqdm(batch_starts, desc="stage3 coverage", total=len(batch_starts)):
            batch = pending_pairs[batch_start : batch_start + args.batch_size]
            golds = [BankClientExtraction.model_validate_json(row["gold_json"]) for row in batch]
            texts = [row["text"] for row in batch]
            coverages = backend.check_coverage_batch(texts, golds)

            for row, coverage in zip(batch, coverages):
                validated_rows.append(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "gold_json": row["gold_json"],
                        "coverage_ok": coverage.all_present,
                        "missing_fields": coverage.missing_fields,
                    }
                )
            save_jsonl(validated_rows, validated_path)
        print(f"stage3 complete: {len(validated_rows)} validated pairs")
    else:
        print(f"stage3 resume: {len(validated_rows)} validated pairs")

    if backend is not None:
        backend.unload()

    test_frame = pd.DataFrame(validated_rows)
    test_frame.to_csv(out_dir / "test.csv", index=False)
    save_jsonl(validated_rows, out_dir / "test.jsonl")
    print(f"Saved {len(validated_rows)} pairs to {out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
