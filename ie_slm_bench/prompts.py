from __future__ import annotations

import json

from pydantic import BaseModel

from schemas.nerel import NerelExtraction
from schemas.runne import RunneExtraction


SYSTEM_PROMPT = (
    "Ты извлекаешь структурированную информацию из русского текста. "
    "Возвращай только JSON, строго соответствующий схеме. "
    "Если поле отсутствует в тексте, оставь пустой список или null. "
    "Не выдумывай значения."
)


def schema_prompt(model_cls: type[BaseModel]) -> str:
    schema = model_cls.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_user_prompt(text: str, model_cls: type[BaseModel]) -> str:
    return (
        "Извлеки структурированную информацию из текста.\n\n"
        f"JSON schema:\n{schema_prompt(model_cls)}\n\n"
        f"Текст:\n{text}"
    )


def benchmark_schema(benchmark: str) -> type[BaseModel]:
    if benchmark == "nerel":
        return NerelExtraction
    if benchmark == "runne":
        return RunneExtraction
    raise ValueError(f"Unknown benchmark: {benchmark}")
