from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from ie_slm_bench.config import BENCHMARK, DATA_DIR, DATASET_REPO, MAX_SAMPLES, SEED
from schemas.bank_client import BankClientExtraction


def subsample_frame(frame: pd.DataFrame, seed: int = SEED, max_samples: int = MAX_SAMPLES) -> pd.DataFrame:
    if len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    sampled = frame.sample(n=max_samples, random_state=seed)
    return sampled.sort_values("doc_id").reset_index(drop=True)


def _load_local_frame(data_dir: Path) -> pd.DataFrame:
    jsonl_path = data_dir / "test.jsonl"
    csv_path = data_dir / "test.csv"
    if jsonl_path.exists():
        rows = []
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append(
                    {
                        "benchmark": BENCHMARK,
                        "doc_id": row["id"],
                        "text": row["text"],
                        "gold_json": row["gold_json"],
                    }
                )
        return pd.DataFrame(rows)
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        return pd.DataFrame(
            {
                "benchmark": BENCHMARK,
                "doc_id": frame["id"],
                "text": frame["text"],
                "gold_json": frame["gold_json"],
            }
        )
    raise FileNotFoundError(f"No dataset found in {data_dir}")


def _load_hf_frame() -> pd.DataFrame:
    path = Path(
        hf_hub_download(
            DATASET_REPO,
            "test.jsonl",
            repo_type="dataset",
        )
    )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "benchmark": BENCHMARK,
                    "doc_id": row["id"],
                    "text": row["text"],
                    "gold_json": row["gold_json"],
                }
            )
    return pd.DataFrame(rows)


def load_gold_frame() -> pd.DataFrame:
    if DATA_DIR.exists() and ((DATA_DIR / "test.jsonl").exists() or (DATA_DIR / "test.csv").exists()):
        frame = _load_local_frame(DATA_DIR)
    else:
        frame = _load_hf_frame()
    records = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="ru-bank-ie gold parsing"):
        gold = BankClientExtraction.model_validate_json(row["gold_json"])
        records.append(
            {
                "benchmark": BENCHMARK,
                "doc_id": int(row["doc_id"]),
                "text": row["text"],
                "gold_json": gold.model_dump_json(by_alias=True, exclude_none=False),
            }
        )
    return subsample_frame(pd.DataFrame(records))


def build_hf_dataset(data_dir: Path) -> Dataset:
    frame = _load_local_frame(data_dir)
    return Dataset.from_pandas(frame)
