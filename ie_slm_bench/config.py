from __future__ import annotations

import os
from pathlib import Path

from ie_slm_bench.env import load_env


load_env()


SEED = int(os.environ.get("IE_SLM_SEED", "42"))
MAX_SAMPLES = int(os.environ.get("IE_SLM_MAX_SAMPLES", "5000"))
MAX_NEW_TOKENS = int(os.environ.get("IE_SLM_MAX_NEW_TOKENS", "1024"))
MAX_INPUT_CHARS = int(os.environ.get("IE_SLM_MAX_INPUT_CHARS", "6000"))
LOAD_IN_4BIT = os.environ.get("IE_SLM_LOAD_IN_4BIT", "1") == "1"
SAVE_EVERY_N = int(os.environ.get("IE_SLM_SAVE_EVERY_N", "1"))
RUN_DIR = Path(os.environ.get("IE_SLM_RUN_DIR", "results/run"))

NEREL_DATASET = os.environ.get("IE_SLM_NEREL_DATASET", "iluvvatar/NEREL")
RUNNE_DATASET = os.environ.get("IE_SLM_RUNNE_DATASET", "iluvvatar/RuNNE")

GEMMA_E2B = os.environ.get("IE_SLM_GEMMA_ID", "google/gemma-4-E2B-it")
QWEN3_17B = os.environ.get("IE_SLM_QWEN3_ID", "Qwen/Qwen3-1.7B")
OLAVA_EXTRACT = os.environ.get("IE_SLM_OLAVA_ID", "numind/NuExtract-2.0-2B")
TINY_PAL = os.environ.get("IE_SLM_TINY_PAL_ID", "LiquidAI/LFM2-1.2B-Extract")

ALL_MODEL_IDS = [
    GEMMA_E2B,
    QWEN3_17B,
    OLAVA_EXTRACT,
    TINY_PAL,
]

DEFAULT_MODEL_IDS = ALL_MODEL_IDS

MODEL_PARAMS = {
    GEMMA_E2B: "2.3B E2B",
    QWEN3_17B: "1.7B",
    OLAVA_EXTRACT: "2B MoE IE",
    TINY_PAL: "1.2B Extract",
}

MODEL_DISPLAY = {
    GEMMA_E2B: "google/gemma-4-e2b",
    QWEN3_17B: "Qwen/Qwen3-1.7B",
    OLAVA_EXTRACT: "olava-extract",
    TINY_PAL: "tiny-pal",
}

DEFAULT_BATCH_SIZES = {
    GEMMA_E2B: 4,
    QWEN3_17B: 8,
    OLAVA_EXTRACT: 8,
    TINY_PAL: 8,
}

GITHUB_REPO = "pymlex/ie-slm-bench"
