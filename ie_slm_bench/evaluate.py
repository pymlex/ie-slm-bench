from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ie_slm_bench.config import RUN_DIR
from ie_slm_bench.data import load_benchmark_frame
from ie_slm_bench.metrics import (
    aggregate_metrics,
    evaluate_predictions,
    safe_model_filename,
)
from ie_slm_bench.models.registry import get_backend
from ie_slm_bench.prompts import benchmark_schema


def run_benchmark_for_model(
    model_id: str,
    benchmark: str,
    run_dir: Path,
    max_new_tokens: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark_dir = run_dir / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_model_filename(model_id)

    gold_path = benchmark_dir / "gold.csv"
    if not gold_path.exists():
        gold_frame = load_benchmark_frame(benchmark)
        gold_frame.to_csv(gold_path, index=False)

    gold_frame = pd.read_csv(gold_path)
    pred_path = benchmark_dir / f"pred_{safe_name}.csv"
    backend = get_backend(model_id)
    print("=" * 80)
    print(f"Loading {model_id} for {benchmark}")
    started = time.time()
    backend.load()
    pred_frame = backend.predict_frame(
        gold_frame,
        benchmark=benchmark,
        max_new_tokens=max_new_tokens,
        pred_path=pred_path,
    )
    backend.unload()
    pred_frame.to_csv(pred_path, index=False)
    print(
        f"Saved {len(pred_frame)} predictions to {pred_path} "
        f"in {time.time() - started:.1f}s"
    )

    model_cls = benchmark_schema(benchmark)
    per_example, per_label = evaluate_predictions(
        pred_frame,
        model_cls=model_cls,
        benchmark=benchmark,
        model_id=model_id,
    )
    per_example_path = benchmark_dir / f"metrics_example_{safe_name}.csv"
    per_label_path = benchmark_dir / f"metrics_label_{safe_name}.csv"
    per_example.to_csv(per_example_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    summary = aggregate_metrics(per_example, per_label)
    summary_path = benchmark_dir / f"metrics_summary_{safe_name}.csv"
    summary.to_csv(summary_path, index=False)
    return pred_frame, per_example, summary


def run_models(
    model_ids: list[str],
    benchmarks: list[str],
    run_dir: Path | None = None,
    max_new_tokens: int = 4096,
) -> pd.DataFrame:
    out_dir = run_dir or RUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for benchmark in benchmarks:
        for model_id in model_ids:
            _, _, summary = run_benchmark_for_model(
                model_id=model_id,
                benchmark=benchmark,
                run_dir=out_dir,
                max_new_tokens=max_new_tokens,
            )
            summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)
