from __future__ import annotations

import json

from schemas.bank_client import BankClientExtraction


SYSTEM_PROMPT = (
    "Ты извлекаешь структурированные данные клиента банка из русского текста. "
    "Верни JSON строго по схеме. "
    "Если поле отсутствует в тексте, оставь null. "
    "Не выдумывай значения."
)

SCHEMA_HINT = (
    '{"Фамилия":"Иванов","Имя":"Иван","Пол":"м","ИНН":"7707083893",'
    '"Адрес регистрации":{"город":"Москва","улица":"Тверская","дом":"1"}}'
)


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_user_prompt(text: str, max_chars: int) -> str:
    clipped = truncate_text(text, max_chars)
    return (
        "Извлеки данные клиента банка из текста.\n\n"
        f"JSON schema example:\n{SCHEMA_HINT}\n\n"
        f"Текст:\n{clipped}"
    )


def output_schema() -> type[BankClientExtraction]:
    return BankClientExtraction
