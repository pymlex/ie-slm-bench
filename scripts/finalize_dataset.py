from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from dataset_gen.finalize import finalize_dataset
from ie_slm_bench.config import DATASET_REPO, DATA_DIR


STAGE_FILES = [
    "stage1_gold.jsonl",
    "stage2_pairs.jsonl",
    "stage3_validated.jsonl",
    "test.jsonl",
    "test.csv",
]


def download_hf_files(data_dir: Path, repo_id: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in STAGE_FILES:
        local_path = data_dir / filename
        if local_path.exists():
            continue
        downloaded = hf_hub_download(
            repo_id,
            filename,
            repo_type="dataset",
        )
        local_path.write_text(Path(downloaded).read_text(encoding="utf-8"), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--repo-id", type=str, default=DATASET_REPO)
    parser.add_argument(
        "--download-hf",
        action="store_true",
        help="Download missing stage files from Hugging Face before finalising.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if args.download_hf:
        download_hf_files(data_dir, args.repo_id)
    stage3_count, test_count = finalize_dataset(data_dir)
    print(f"stage3_validated.jsonl: {stage3_count} rows")
    print(f"test.jsonl: {test_count} coverage-valid rows")


if __name__ == "__main__":
    main()
