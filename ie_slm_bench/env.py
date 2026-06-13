from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    load_dotenv(ROOT / ".env")
