from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from ie_slm_bench.benchmark_summary import benchmark_results_markdown
from ie_slm_bench.config import DATA_DIR, DATASET_REPO


def build_dataset_card(results_root: Path) -> str:
    benchmark_section = benchmark_results_markdown(results_root)
    if not benchmark_section:
        benchmark_section = (
            "Results are published in the GitHub repository "
            "`pymlex/ie-slm-bench` under `results/`."
        )
    return DATASET_CARD_TEMPLATE.replace("{BENCHMARK_SECTION}", benchmark_section)


DATASET_CARD_TEMPLATE = """---
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

Russian bank client information extraction benchmark with coverage-validated text-to-JSON pairs.
Each example contains a chat-style client message, a gold `BankClientExtraction` JSON object,
and a separate `validation_json` coverage justification. Fields may be `null` when absent from the source text.

## Columns

- `id` — sample identifier
- `reasoning` — model planning before the client message
- `text` — client message used for evaluation
- `gold_json` — gold `BankClientExtraction` JSON
- `validation_json` — stage-3 coverage check with `all_present`, `missing_fields`, `justification`

## Schema

Fixed Pydantic schema in `schemas/bank_client.py` with Russian field aliases.

## Files

- `test.jsonl` — evaluation split, coverage-valid rows only
- `test.csv` — same data in CSV
- `stage1_gold.jsonl` — gold JSON profiles
- `stage2_pairs.jsonl` — text prompts from gold JSON
- `stage3_validated.jsonl` — all stage-2 pairs with `validation_json`

## Benchmark results

Evaluation on $N=368$ coverage-valid pairs with three SLMs up to 2B parameters on NVIDIA RTX 5090.
Metrics use normalised string comparison for phones, INN, passport codes, dates and case-folded text fields.

{BENCHMARK_SECTION}

## Generation

Dataset built with `Qwen/Qwen3.5-4B` and Outlines constrained decoding on Colab.

```bash
bash scripts/generate_dataset.sh --n 500
bash scripts/finalize_and_push_dataset.sh --data-dir data/ru-bank-ie
```

## Citation

```bibtex
@misc{zyukov2026ru_bank_ie,
  title={ru-bank-ie: Russian Bank Client Information Extraction Benchmark},
  author={Zyukov, Alexey},
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
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Upload README.md card only without dataset files.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    card_text = build_dataset_card(repo_root / "results")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    if args.card_only:
        assets_src = repo_root / "results" / "assets"
        upload_paths: list[tuple[str, bytes]] = [("README.md", card_text.encode("utf-8"))]
        if assets_src.exists():
            for png in sorted(assets_src.glob("*.png")):
                upload_paths.append(
                    (f"benchmark_assets/{png.name}", png.read_bytes())
                )
        for path_in_repo, payload in upload_paths:
            api.upload_file(
                path_or_fileobj=payload,
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Update ru-bank-ie dataset card with benchmark analysis",
            )
        print(f"Updated card at https://huggingface.co/datasets/{args.repo_id}")
        return

    data_dir = args.data_dir
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}. Run dataset generation first.")

    card_path = data_dir / "README.md"
    card_path.write_text(card_text, encoding="utf-8")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload ru-bank-ie benchmark dataset",
    )
    print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
