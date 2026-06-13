#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -q -r requirements.txt

if ! command -v gh >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq gh
fi

if ! gh auth status >/dev/null 2>&1; then
  gh auth login --web --git-protocol https
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
