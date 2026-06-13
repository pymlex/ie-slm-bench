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


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    skeleton_path = out_dir / "stage1_skeletons.jsonl"
    gold_path = out_dir / "stage2_gold.jsonl"
    pair_path = out_dir / "stage3_pairs.jsonl"
    validated_path = out_dir / "stage4_validated.jsonl"

    skeleton_rows = load_jsonl(skeleton_path)
    if len(skeleton_rows) != args.n:
        skeleton_rows = [build_skeleton(rng, sample_id=index) for index in range(args.n)]
        save_jsonl(skeleton_rows, skeleton_path)

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
        batch_starts = range(len(gold_rows), args.n, args.batch_size)
        for batch_start in tqdm(batch_starts, desc="stage2 gold", total=len(batch_starts)):
            batch = skeleton_rows[batch_start : batch_start + args.batch_size]
            persons = backend.generate_person_fill_batch(batch)
            for skeleton, person in zip(batch, persons):
                gold = merge_skeleton_and_person(skeleton, person)
                gold_rows.append(
                    {
                        "id": skeleton["sample_id"],
                        "gold_json": gold.model_dump_json(by_alias=True, exclude_none=False),
                    }
                )
            save_jsonl(gold_rows, gold_path)
        print(f"stage2 complete: {len(gold_rows)} gold rows")
    else:
        print(f"stage2 resume: {len(gold_rows)} gold rows")

    if len(pair_rows) < args.n:
        pending_gold = gold_rows[len(pair_rows) :]
        batch_starts = range(0, len(pending_gold), args.batch_size)
        for batch_start in tqdm(batch_starts, desc="stage3 texts", total=len(batch_starts)):
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
        print(f"stage3 complete: {len(pair_rows)} pairs")
    else:
        print(f"stage3 resume: {len(pair_rows)} pairs")

    if len(validated_rows) < args.n:
        pending_pairs = pair_rows[len(validated_rows) :]
        batch_starts = range(0, len(pending_pairs), args.batch_size)
        for batch_start in tqdm(batch_starts, desc="stage4 coverage", total=len(batch_starts)):
            batch = pending_pairs[batch_start : batch_start + args.batch_size]
            golds = [BankClientExtraction.model_validate_json(row["gold_json"]) for row in batch]
            texts = [row["text"] for row in batch]
            coverages = backend.check_coverage_batch(texts, golds)

            retry_indices = [
                index
                for index, coverage in enumerate(coverages)
                if not coverage.all_present
            ]
            if retry_indices:
                retry_golds = [golds[index] for index in retry_indices]
                retry_texts = backend.generate_text_batch(retry_golds)
                for local_index, global_index in enumerate(retry_indices):
                    texts[global_index] = retry_texts[local_index]
                retry_coverages = backend.check_coverage_batch(
                    [texts[index] for index in retry_indices],
                    retry_golds,
                )
                for local_index, global_index in enumerate(retry_indices):
                    coverages[global_index] = retry_coverages[local_index]

            for row, text, coverage in zip(batch, texts, coverages):
                validated_rows.append(
                    {
                        "id": row["id"],
                        "text": text,
                        "gold_json": row["gold_json"],
                        "coverage_ok": coverage.all_present,
                        "missing_fields": coverage.missing_fields,
                    }
                )
            save_jsonl(validated_rows, validated_path)
        print(f"stage4 complete: {len(validated_rows)} validated pairs")
    else:
        print(f"stage4 resume: {len(validated_rows)} validated pairs")

    if backend is not None:
        backend.unload()

    test_frame = pd.DataFrame(validated_rows)
    test_frame.to_csv(out_dir / "test.csv", index=False)
    save_jsonl(validated_rows, out_dir / "test.jsonl")
    print(f"Saved {len(validated_rows)} pairs to {out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
