from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from ie_slm_bench.benchmark_summary import benchmark_results_markdown
from ie_slm_bench.config import DATA_DIR, DATASET_REPO, RUN_DIR
from ie_slm_bench.hf_benchmark import iter_staged_files, stage_benchmark_tree


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
- `benchmark/runs/` — per-model metrics CSV artefacts
- `benchmark/results.json` — aggregated benchmark scores
- `benchmark/metrics.json` — summary metrics JSON from GitHub `results/`
- `benchmark/analysis.json` — per-field and per-group breakdown
- `benchmark/summary.csv` — metrics table
- `benchmark_assets/` — benchmark plots

## Benchmark results

Evaluation on the full coverage-valid split ($N=368$) with three SLMs up to 2B parameters on NVIDIA RTX 5090.
Metrics use normalised string comparison for phones, INN, passport codes, dates and case-folded text fields.
The split is built from 500 stage-3 rows: only pairs with `all_present=true` and non-empty `text` are retained.

{BENCHMARK_SECTION}

## Generation

Dataset built with `Qwen/Qwen3.5-4B` and Outlines constrained decoding on Colab.
Stage 1 generates 500 gold profiles with random field sparsity in $[0.2, 0.8]$.
Stage 2 produces client messages split into `reasoning` and `text`.
Stage 3 validates field coverage on `text` only.

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

DEPRECATED_LM_EVAL_CARD = """---
language:
- ru
license: gpl-3.0
---

# pymlex/ru-bank-ie-lm-eval

This repository is deprecated. All dataset and benchmark artefacts now live in
[pymlex/ru-bank-ie](https://huggingface.co/datasets/pymlex/ru-bank-ie).

Benchmark metrics: `benchmark/` and `benchmark_assets/` in the main dataset repository.
"""


def upload_benchmark_and_card(
    api: HfApi,
    repo_id: str,
    card_text: str,
    staging_root: Path,
    commit_message: str,
) -> None:
    upload_paths = iter_staged_files(staging_root)
    upload_paths.append(("README.md", card_text.encode("utf-8")))
    for path_in_repo, payload in upload_paths:
        api.upload_file(
            path_or_fileobj=payload,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--repo-id", type=str, default=DATASET_REPO)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Upload README.md and benchmark artefacts without dataset files.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Alias for --card-only.",
    )
    parser.add_argument(
        "--deprecate-legacy-repo",
        action="store_true",
        help="Upload deprecation notice to pymlex/ru-bank-ie-lm-eval.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "results"
    card_text = build_dataset_card(results_root)
    card_only = args.card_only or args.metrics_only

    staging_root = results_root / "hf-staging"
    stage_benchmark_tree(staging_root, args.run_dir, results_root)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    if card_only:
        upload_benchmark_and_card(
            api,
            args.repo_id,
            card_text,
            staging_root,
            "Update ru-bank-ie card and benchmark artefacts",
        )
        print(f"Updated https://huggingface.co/datasets/{args.repo_id}")
    else:
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
            commit_message="Upload ru-bank-ie dataset",
        )
        upload_benchmark_and_card(
            api,
            args.repo_id,
            card_text,
            staging_root,
            "Upload ru-bank-ie benchmark artefacts",
        )
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")

    if args.deprecate_legacy_repo:
        legacy_repo = "pymlex/ru-bank-ie-lm-eval"
        api.create_repo(legacy_repo, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=DEPRECATED_LM_EVAL_CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=legacy_repo,
            repo_type="dataset",
            commit_message="Deprecate repo, metrics moved to pymlex/ru-bank-ie",
        )
        print(f"Deprecated https://huggingface.co/datasets/{legacy_repo}")

    if staging_root.exists():
        shutil.rmtree(staging_root)


if __name__ == "__main__":
    main()
