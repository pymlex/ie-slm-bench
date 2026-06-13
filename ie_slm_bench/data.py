from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from ie_slm_bench.config import MAX_SAMPLES, NEREL_DATASET, RUNNE_DATASET, SEED
from ie_slm_bench.parsers import parse_nerel_gold, parse_runne_gold


def _download_jsonl(dataset_id: str, split: str) -> Path:
    return Path(
        hf_hub_download(
            dataset_id,
            f"data/{split}.jsonl",
            repo_type="dataset",
        )
    )


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _load_nerel_rows() -> list[dict]:
    rows = []
    for split in ("train", "test", "dev"):
        rows.extend(_load_jsonl_rows(_download_jsonl(NEREL_DATASET, split)))
    return rows


def _load_runne_rows() -> list[dict]:
    rows = []
    for split in ("train", "test"):
        path = _download_jsonl(RUNNE_DATASET, split)
        split_rows = _load_jsonl_rows(path)
        rows.extend(split_rows)
    return rows


def subsample_rows(rows: list[dict], seed: int = SEED, max_samples: int = MAX_SAMPLES) -> list[dict]:
    if len(rows) <= max_samples:
        return rows
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=max_samples, replace=False)
    indices = np.sort(indices)
    return [rows[index] for index in indices]


def build_nerel_frame() -> pd.DataFrame:
    rows = subsample_rows(_load_nerel_rows())
    records = []
    for row in tqdm(rows, desc="NEREL gold parsing"):
        gold = parse_nerel_gold(row)
        records.append(
            {
                "benchmark": "nerel",
                "doc_id": row["id"],
                "text": row["text"],
                "gold_json": gold.model_dump_json(),
            }
        )
    return pd.DataFrame(records)


def build_runne_frame() -> pd.DataFrame:
    rows = subsample_rows(_load_runne_rows())
    records = []
    for row in tqdm(rows, desc="RuNNE gold parsing"):
        gold = parse_runne_gold(row)
        records.append(
            {
                "benchmark": "runne",
                "doc_id": row["id"],
                "text": row["text"],
                "gold_json": gold.model_dump_json(),
            }
        )
    return pd.DataFrame(records)


def load_benchmark_frame(benchmark: str) -> pd.DataFrame:
    if benchmark == "nerel":
        return build_nerel_frame()
    if benchmark == "runne":
        return build_runne_frame()
    raise ValueError(f"Unknown benchmark: {benchmark}")
