from __future__ import annotations

from ie_slm_bench.config import (
    ALL_MODEL_IDS,
    DEFAULT_BATCH_SIZES,
    GEMMA_E2B,
    OLAVA_EXTRACT,
    QWEN3_17B,
    TINY_PAL,
)
from ie_slm_bench.models.structured_lm import StructuredLmBackend


def get_backend(model_id: str) -> StructuredLmBackend:
    batch_size = DEFAULT_BATCH_SIZES.get(model_id, 4)
    if model_id == GEMMA_E2B:
        return StructuredLmBackend(model_id, batch_size=batch_size, backend_kind="gemma")
    if model_id == OLAVA_EXTRACT:
        return StructuredLmBackend(model_id, batch_size=batch_size, backend_kind="nuextract")
    if model_id == TINY_PAL:
        return StructuredLmBackend(model_id, batch_size=batch_size, backend_kind="lfm")
    return StructuredLmBackend(model_id, batch_size=batch_size, backend_kind="causal")


def list_model_ids() -> list[str]:
    return list(ALL_MODEL_IDS)
