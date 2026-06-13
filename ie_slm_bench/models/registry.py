from __future__ import annotations

from ie_slm_bench.config import (
    ALL_MODEL_IDS,
    DEFAULT_BATCH_SIZES,
    OLAVA_EXTRACT,
    QWEN3_17B,
    TINY_PAL,
)
from ie_slm_bench.models.nuextract_lm import NuExtractBackend
from ie_slm_bench.models.structured_lm import StructuredLmBackend


def get_backend(model_id: str) -> StructuredLmBackend | NuExtractBackend:
    batch_size = DEFAULT_BATCH_SIZES.get(model_id, 8)
    if model_id == OLAVA_EXTRACT:
        return NuExtractBackend(model_id, batch_size=batch_size)
    if model_id == TINY_PAL:
        return StructuredLmBackend(model_id, batch_size=batch_size, backend_kind="lfm")
    return StructuredLmBackend(model_id, batch_size=batch_size, backend_kind="causal")


def list_model_ids() -> list[str]:
    return list(ALL_MODEL_IDS)
