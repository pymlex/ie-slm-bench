from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from ie_slm_bench.config import DATA_DIR, DATASET_REPO


DATASET_CARD = """---
language:
- ru
license: gpl-3.0
task_categories:
- text-generation
- question-answering
tags:
- information-extraction
- banking
- russian
- pydantic
size_categories:
- n<1K
---

# pymlex/ru-bank-ie

Russian bank client information extraction benchmark with 500 text-to-JSON pairs.
Each example contains a chat-style client message and a gold `BankClientExtraction` JSON object.
Fields may be `null` when absent from the source text.

## Schema

Fixed Pydantic schema in `schemas/bank_client.py` with Russian field aliases.

## Files

- `test.jsonl` — evaluation split
- `test.csv` — same data in CSV
- `stage1_gold.jsonl` — gold JSON profiles
- `stage2_pairs.jsonl` — text prompts from gold JSON
- `stage3_validated.jsonl` — coverage-checked pairs

## Generation

Dataset built with `Qwen/Qwen3.5-4B` and Outlines constrained decoding on Colab.

```bash
bash scripts/generate_dataset.sh --n 500
python scripts/push_dataset_hf.py
```

## Citation

```bibtex
@misc{zyukov2026ru_bank_ie,
  title={ru-bank-ie: Russian Bank Client Information Extraction Benchmark},
  author={Zyukov, Aleksandr and pymlex},
  year={2026},
  howpublished={\\url{https://huggingface.co/datasets/pymlex/ru-bank-ie}}
}
```

The project is under GPL-3.0 license.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--repo-id", type=str, default=DATASET_REPO)
    args = parser.parse_args()

    data_dir = args.data_dir
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}. Run dataset generation first.")

    card_path = data_dir / "README.md"
    card_path.write_text(DATASET_CARD, encoding="utf-8")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload ru-bank-ie benchmark dataset",
    )
    print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
