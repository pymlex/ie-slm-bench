#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f .env ]; then
  cp .env.example .env
fi

export TORCHDYNAMO_DISABLE=1

python scripts/finalize_dataset.py "$@"
python scripts/push_dataset_hf.py --data-dir "${IE_SLM_DATA_DIR:-data/ru-bank-ie}"
