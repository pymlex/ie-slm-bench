from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def configure_torch_runtime() -> None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def load_env() -> None:
    configure_torch_runtime()
    load_dotenv(ROOT / ".env")
