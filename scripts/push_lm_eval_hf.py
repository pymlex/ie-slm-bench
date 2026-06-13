from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from ie_slm_bench.config import BENCHMARK, LM_EVAL_REPO, RUN_DIR
from ie_slm_bench.metrics import safe_model_filename


LM_EVAL_CARD = """---
language:
- ru
license: gpl-3.0
task_categories:
- text-generation
tags:
- lm-eval
- information-extraction
- banking
---

# pymlex/ru-bank-ie-lm-eval

Evaluation artefacts for the `pymlex/ru-bank-ie` benchmark.
Metrics are stored per model in `runs/` and aggregated in `results.json`.

## Metrics

- `strict_exact_match`
- `field_precision`, `field_recall`, `field_f1`
- `null_field_accuracy`
- `hallucination_rate`
- `schema_validity_rate`
- `entity_f1`

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


def build_lm_eval_results(summary: pd.DataFrame) -> dict:
    results: dict[str, dict[str, float]] = {}
    for _, row in summary.iterrows():
        model_key = safe_model_filename(row["model_id"])
        results[model_key] = {
            f"strict_exact_match,{BENCHMARK}": float(row["strict_exact_match"]),
            f"field_precision,{BENCHMARK}": float(row["field_precision"]),
            f"field_recall,{BENCHMARK}": float(row["field_recall"]),
            f"field_f1,{BENCHMARK}": float(row["field_f1"]),
            f"null_field_accuracy,{BENCHMARK}": float(row["null_field_accuracy"]),
            f"hallucination_rate,{BENCHMARK}": float(row["hallucination_rate"]),
            f"schema_validity_rate,{BENCHMARK}": float(row["schema_validity_rate"]),
            f"entity_f1,{BENCHMARK}": float(row["entity_f1"]),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--repo-id", type=str, default=LM_EVAL_REPO)
    parser.add_argument("--out-dir", type=Path, default=Path("results/lm-eval-upload"))
    args = parser.parse_args()

    summary_frames = []
    for path in args.run_dir.glob("metrics_summary_*.csv"):
        summary_frames.append(pd.read_csv(path))
    if not summary_frames:
        raise FileNotFoundError(f"No metrics_summary_*.csv in {args.run_dir}")

    summary = pd.concat(summary_frames, ignore_index=True)
    upload_dir = args.out_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / "README.md").write_text(LM_EVAL_CARD, encoding="utf-8")

    runs_dir = upload_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for path in args.run_dir.glob("metrics_*"):
        destination = runs_dir / path.name
        destination.write_bytes(path.read_bytes())

    metrics_path = args.run_dir.parent / "metrics.json"
    if metrics_path.exists():
        (upload_dir / "metrics.json").write_bytes(metrics_path.read_bytes())

    payload = {
        "benchmark": BENCHMARK,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "results": build_lm_eval_results(summary),
        "summary": summary.to_dict(orient="records"),
    }
    (upload_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary.to_csv(upload_dir / "summary.csv", index=False)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload ru-bank-ie lm-eval metrics",
    )
    print(f"Pushed lm-eval results to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
