from __future__ import annotations

from schemas.runne import RunneExtraction


SYSTEM_PROMPT = (
    "Ты извлекаешь структурированную информацию из русского текста. "
    "Возвращай только JSON, строго соответствующий схеме. "
    "Если поле отсутствует в тексте, оставь пустой список или null. "
    "Не выдумывай значения."
)

RUNNE_SCHEMA_HINT = '{"entities":[{"start":0,"end":7,"type":"PERSON"}]}'


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_user_prompt(text: str, max_chars: int) -> str:
    clipped = truncate_text(text, max_chars)
    return (
        "Извлеки структурированную информацию из текста.\n\n"
        f"JSON schema example:\n{RUNNE_SCHEMA_HINT}\n\n"
        f"Текст:\n{clipped}"
    )


def output_schema() -> type[RunneExtraction]:
    return RunneExtraction
