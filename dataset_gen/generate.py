from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dataset_gen.llm import GeneratorBackend, merge_skeleton_and_person
from dataset_gen.skeleton import build_skeleton
from ie_slm_bench.config import DATASET_SIZE, DATA_DIR, GEN_BATCH_SIZE, SEED
from schemas.bank_client import BankClientExtraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DATASET_SIZE)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=GEN_BATCH_SIZE)
    return parser.parse_args()


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    skeleton_rows = [build_skeleton(rng, sample_id=index) for index in range(args.n)]
    save_jsonl(skeleton_rows, out_dir / "stage1_skeletons.jsonl")

    backend = GeneratorBackend()
    backend.load()
    gold_rows = []
    for skeleton in tqdm(skeleton_rows, desc="stage2 gold"):
        person = backend.generate_person_fill(skeleton)
        gold = merge_skeleton_and_person(skeleton, person)
        gold_rows.append(
            {
                "id": skeleton["sample_id"],
                "gold_json": gold.model_dump_json(by_alias=True, exclude_none=False),
            }
        )
    save_jsonl(gold_rows, out_dir / "stage2_gold.jsonl")

    pair_rows = []
    batch_starts = range(0, len(gold_rows), args.batch_size)
    for batch_start in tqdm(batch_starts, desc="stage3 texts"):
        batch = gold_rows[batch_start : batch_start + args.batch_size]
        for row in batch:
            gold = BankClientExtraction.model_validate_json(row["gold_json"])
            text = backend.generate_text(gold)
            pair_rows.append(
                {
                    "id": row["id"],
                    "text": text,
                    "gold_json": row["gold_json"],
                }
            )
        save_jsonl(pair_rows, out_dir / "stage3_pairs.jsonl")

    validated_rows = []
    for row in tqdm(pair_rows, desc="stage4 coverage"):
        gold = BankClientExtraction.model_validate_json(row["gold_json"])
        coverage = backend.check_coverage(row["text"], gold)
        text = row["text"]
        if not coverage.all_present:
            text = backend.generate_text(gold)
            coverage = backend.check_coverage(text, gold)
        validated_rows.append(
            {
                "id": row["id"],
                "text": text,
                "gold_json": row["gold_json"],
                "coverage_ok": coverage.all_present,
                "missing_fields": coverage.missing_fields,
            }
        )
    save_jsonl(validated_rows, out_dir / "stage4_validated.jsonl")

    backend.unload()

    test_frame = pd.DataFrame(validated_rows)
    test_frame.to_csv(out_dir / "test.csv", index=False)
    save_jsonl(validated_rows, out_dir / "test.jsonl")
    print(f"Saved {len(validated_rows)} pairs to {out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
