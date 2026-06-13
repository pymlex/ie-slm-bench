from __future__ import annotations

import argparse
import json
from pathlib import Path

import pynvml
import torch

from ie_slm_bench.config import ALL_MODEL_IDS, DEFAULT_MODEL_IDS, MAX_NEW_TOKENS, RUN_DIR
from ie_slm_bench.evaluate import run_models
from ie_slm_bench.plots import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SLMs on Russian structured information extraction",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_IDS,
        help="HuggingFace model ids",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["nerel", "runne"],
        choices=["nerel", "runne"],
        help="Benchmarks to evaluate separately",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(RUN_DIR),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all registered models",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Rebuild plots from existing CSV files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pynvml.nvmlInit()
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: none")

    if args.plots_only:
        summary = generate_all_plots(run_dir, run_dir.parent / "assets")
    else:
        model_ids = ALL_MODEL_IDS if args.all_models else args.models
        summary = run_models(
            model_ids=model_ids,
            benchmarks=args.benchmarks,
            run_dir=run_dir,
            max_new_tokens=args.max_new_tokens,
        )
        summary = generate_all_plots(run_dir, run_dir.parent / "assets")

    metrics_path = run_dir.parent / "metrics.json"
    metrics_path.write_text(
        json.dumps({"summary": summary.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("Wrote", metrics_path)


if __name__ == "__main__":
    main()
