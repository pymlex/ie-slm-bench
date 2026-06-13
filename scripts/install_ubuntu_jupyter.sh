#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f .env ]; then
  cp .env.example .env
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq build-essential
fi

pip install -q -r requirements.txt

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
