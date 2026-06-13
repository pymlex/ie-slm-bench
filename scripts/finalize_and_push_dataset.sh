#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f .env ]; then
  cp .env.example .env
fi

export TORCHDYNAMO_DISABLE=1

DATA_DIR="data/ru-bank-ie"
prev=""
for arg in "$@"; do
  if [ "$prev" = "--data-dir" ]; then
    DATA_DIR="$arg"
  fi
  prev="$arg"
done

python scripts/finalize_dataset.py "$@"
python scripts/push_dataset_hf.py --data-dir "$DATA_DIR"
