#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python scripts/setup_gh_auth.py
python main.py --all-models --run-dir results/run
python scripts/push_lm_eval_hf.py --run-dir results/run
python scripts/push_results_github.py --message "Colab: IE SLM benchmark results"
