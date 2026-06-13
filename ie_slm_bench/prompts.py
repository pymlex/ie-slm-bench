from __future__ import annotations

import json

from pydantic import BaseModel

from ie_slm_bench.config import MAX_INPUT_CHARS
from schemas.nerel import NerelExtraction
from schemas.runne import RunneExtraction


SYSTEM_PROMPT = (
    "Ты извлекаешь структурированную информацию из русского текста. "
    "Возвращай только JSON, строго соответствующий схеме. "
    "Если поле отсутствует в тексте, оставь пустой список или null. "
    "Не выдумывай значения."
)

NEREL_SCHEMA_HINT = (
    '{"entities":[{"id":"T1","type":"PERSON","start":0,"end":5,"text":"..."}],'
    '"relations":[{"id":"R1","type":"WORKPLACE","arg1":"T2","arg2":"T3"}],'
    '"links":[{"id":"N1","entity_id":"T5","reference":"Wikidata:Q1","kb_name":null}]}'
)

RUNNE_SCHEMA_HINT = '{"entities":[{"start":0,"end":7,"type":"PERSON"}]}'


def truncate_text(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def schema_hint(benchmark: str) -> str:
    if benchmark == "nerel":
        return NEREL_SCHEMA_HINT
    if benchmark == "runne":
        return RUNNE_SCHEMA_HINT
    raise ValueError(f"Unknown benchmark: {benchmark}")


def build_user_prompt(text: str, benchmark: str) -> str:
    clipped = truncate_text(text)
    return (
        "Извлеки структурированную информацию из текста.\n\n"
        f"JSON schema example:\n{schema_hint(benchmark)}\n\n"
        f"Текст:\n{clipped}"
    )


def benchmark_schema(benchmark: str) -> type[BaseModel]:
    if benchmark == "nerel":
        return NerelExtraction
    if benchmark == "runne":
        return RunneExtraction
    raise ValueError(f"Unknown benchmark: {benchmark}")
