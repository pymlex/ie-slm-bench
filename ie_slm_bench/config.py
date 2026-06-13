from __future__ import annotations

import os
from pathlib import Path

from ie_slm_bench.env import load_env


load_env()


BENCHMARK = "ru-bank-ie"
SEED = int(os.environ.get("IE_SLM_SEED", "42"))
DATASET_SIZE = int(os.environ.get("IE_SLM_DATASET_SIZE", "500"))
MAX_SAMPLES = int(os.environ.get("IE_SLM_MAX_SAMPLES", "5000"))
MAX_NEW_TOKENS = int(os.environ.get("IE_SLM_MAX_NEW_TOKENS", "512"))
MAX_INPUT_CHARS = int(os.environ.get("IE_SLM_MAX_INPUT_CHARS", "6000"))
MAX_INPUT_TOKENS = int(os.environ.get("IE_SLM_MAX_INPUT_TOKENS", "2048"))
LOAD_IN_4BIT = os.environ.get("IE_SLM_LOAD_IN_4BIT", "0") == "1"
SAVE_EVERY_N = int(os.environ.get("IE_SLM_SAVE_EVERY_N", "1"))
RUN_DIR = Path(os.environ.get("IE_SLM_RUN_DIR", "results/run"))

DATASET_REPO = os.environ.get("IE_SLM_DATASET_REPO", "pymlex/ru-bank-ie")
LM_EVAL_REPO = os.environ.get("IE_SLM_LM_EVAL_REPO", "pymlex/ru-bank-ie-lm-eval")
DATA_DIR = Path(os.environ.get("IE_SLM_DATA_DIR", "data/ru-bank-ie"))
GENERATOR_MODEL = os.environ.get("IE_SLM_GENERATOR_MODEL", "Qwen/Qwen3.5-4B")
GEN_BATCH_SIZE = int(os.environ.get("IE_SLM_GEN_BATCH_SIZE", "8"))

QWEN3_17B = os.environ.get("IE_SLM_QWEN3_ID", "Qwen/Qwen3-1.7B")
OLAVA_EXTRACT = os.environ.get("IE_SLM_OLAVA_ID", "numind/NuExtract-2.0-2B")
TINY_PAL = os.environ.get("IE_SLM_TINY_PAL_ID", "LiquidAI/LFM2-1.2B-Extract")

ALL_MODEL_IDS = [
    QWEN3_17B,
    OLAVA_EXTRACT,
    TINY_PAL,
]

DEFAULT_MODEL_IDS = ALL_MODEL_IDS

MODEL_DISPLAY = {
    QWEN3_17B: "Qwen/Qwen3-1.7B",
    OLAVA_EXTRACT: "olava-extract",
    TINY_PAL: "tiny-pal",
}

DEFAULT_BATCH_SIZES = {
    QWEN3_17B: int(os.environ.get("IE_SLM_BATCH_SIZE_QWEN3", "16")),
    OLAVA_EXTRACT: int(os.environ.get("IE_SLM_BATCH_SIZE_OLAVA", "12")),
    TINY_PAL: int(os.environ.get("IE_SLM_BATCH_SIZE_TINY_PAL", "24")),
}

GITHUB_REPO = "pymlex/ie-slm-bench"
