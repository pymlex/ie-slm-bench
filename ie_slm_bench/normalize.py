from __future__ import annotations

import re

DIGITS_ONLY = re.compile(r"\D+")
WHITESPACE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    return WHITESPACE.sub(" ", text.strip())


def normalize_phone(value: str) -> str:
    digits = DIGITS_ONLY.sub("", value)
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    if len(digits) == 10:
        digits = "7" + digits
    return digits


def normalize_inn(value: str) -> str:
    return DIGITS_ONLY.sub("", value)


def normalize_snils(value: str) -> str:
    digits = DIGITS_ONLY.sub("", value)
    if len(digits) < 11:
        return digits
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:9]} {digits[9:11]}"


def normalize_passport_series_number(value: str) -> str:
    digits = DIGITS_ONLY.sub("", value)
    if len(digits) >= 10:
        return f"{digits[:4]} {digits[4:10]}"
    return collapse_whitespace(value)


def normalize_department_code(value: str) -> str:
    digits = DIGITS_ONLY.sub("", value)
    if len(digits) >= 6:
        return f"{digits[:3]}-{digits[3:6]}"
    return collapse_whitespace(value)


def normalize_date(value: str) -> str:
    stripped = collapse_whitespace(value)
    match = re.fullmatch(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", stripped)
    if match is None:
        return stripped
    day, month, year = match.groups()
    return f"{int(day):02d}.{int(month):02d}.{year}"


def normalize_gender(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"м", "m", "male", "муж", "мужской"}:
        return "м"
    if lowered in {"ж", "f", "female", "жен", "женский"}:
        return "ж"
    return value.strip()


def normalize_field_value(path: str, value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None

    if path == "ИНН":
        normalized = normalize_inn(stripped)
        return normalized or None
    if path == "СНИЛС":
        normalized = normalize_snils(stripped)
        return normalized or None
    if path == "Номер мобильного телефона":
        normalized = normalize_phone(stripped)
        return normalized or None
    if path == "Серия и номер паспорта":
        return normalize_passport_series_number(stripped)
    if path == "Код подразделения":
        return normalize_department_code(stripped)
    if path in {"Дата рождения", "Дата выдачи паспорта"}:
        return normalize_date(stripped)
    if path == "Пол":
        return normalize_gender(stripped)
    if path == "Адрес электронной почты":
        return stripped.lower()
    if path == "Ежемесячный доход":
        digits = DIGITS_ONLY.sub("", stripped)
        return digits or stripped
    return collapse_whitespace(stripped).casefold()
