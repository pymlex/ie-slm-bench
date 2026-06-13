from __future__ import annotations

import numpy as np

from schemas.bank_client import WorkExperience


GOLD_FIELDS = [
    "surname",
    "name",
    "patronymic",
    "birth_date",
    "birth_year",
    "birth_place",
    "citizenship",
    "gender",
    "passport_series_number",
    "passport_issued_by",
    "passport_issue_date",
    "passport_department_code",
    "inn",
    "snils",
    "registration_address",
    "actual_address",
    "mobile_phone",
    "email",
    "employer",
    "job_title",
    "work_experience",
    "monthly_income",
    "marital_status",
    "dependents_count",
    "real_estate",
    "car",
    "loans_count",
]

REGION_HINTS = np.array(
    [
        "Москва",
        "Санкт-Петербург",
        "Казань",
        "Екатеринбург",
        "Новосибирск",
        "Краснодар",
        "Воронеж",
        "Пермь",
        "Уфа",
        "Тюмень",
        "Самара",
        "Ростов-на-Дону",
        "Владивосток",
        "Иркутск",
        "Калининград",
    ],
    dtype=object,
)

JOB_HINTS = np.array(
    [
        "логистическая компания",
        "строительная организация",
        "IT-компания",
        "медицинский центр",
        "торговая сеть",
        "производственный комбинат",
        "транспортная компания",
        "образовательный центр",
        "банковский филиал",
        "рекламное агентство",
        "фармацевтическая фирма",
        "гостиничный комплекс",
    ],
    dtype=object,
)


def inn_checksum(digits: np.ndarray) -> int:
    weights1 = np.array([7, 2, 4, 10, 3, 5, 9, 4, 6, 8], dtype=np.int64)
    weights2 = np.array([3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8], dtype=np.int64)
    s1 = int(np.sum(digits[:10] * weights1) % 11 % 10)
    tail = np.concatenate([digits[:10], np.array([s1], dtype=np.int64)])
    s2 = int(np.sum(tail * weights2) % 11 % 10)
    return s2


def random_inn(rng: np.random.Generator) -> str:
    digits = rng.integers(0, 10, size=10, dtype=np.int64)
    check = inn_checksum(digits)
    full = np.concatenate([digits, np.array([check], dtype=np.int64)])
    return "".join(str(int(value)) for value in full)


def snils_checksum(digits: np.ndarray) -> int:
    weights = np.arange(9, 0, -1, dtype=np.int64)
    total = int(np.sum(digits * weights))
    if total < 100:
        return total
    if total in (100, 101):
        return 0
    return int(total % 101 % 100)


def random_snils(rng: np.random.Generator) -> str:
    digits = rng.integers(0, 10, size=9, dtype=np.int64)
    check = snils_checksum(digits)
    body = "".join(str(int(value)) for value in digits)
    return f"{body[:3]}-{body[3:6]}-{body[6:9]} {check:02d}"


def random_passport_series_number(rng: np.random.Generator) -> str:
    series = int(rng.integers(0, 10000))
    number = int(rng.integers(0, 1_000_000))
    return f"{series:04d} {number:06d}"


def random_department_code(rng: np.random.Generator) -> str:
    left = int(rng.integers(100, 1000))
    right = int(rng.integers(0, 1000))
    return f"{left:03d}-{right:03d}"


def random_birth_date(rng: np.random.Generator) -> tuple[str, int]:
    year = int(rng.integers(1955, 2006))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 29))
    return f"{day:02d}.{month:02d}.{year}", year


def random_phone(rng: np.random.Generator) -> str:
    tail = int(rng.integers(0, 10_000_000))
    prefix = int(
        rng.choice(
            np.array([903, 905, 906, 909, 915, 916, 917, 919, 925, 926, 929, 977, 999], dtype=np.int64)
        )
    )
    return f"+7{prefix}{tail:07d}"


def random_email(rng: np.random.Generator, token: str) -> str:
    domain = str(rng.choice(np.array(["mail.ru", "yandex.ru", "gmail.com", "inbox.ru"], dtype=object)))
    return f"client_{token}@{domain}"


def field_mask(rng: np.random.Generator, fields: list[str], keep_ratio: float) -> dict[str, bool]:
    flags = rng.random(len(fields)) < keep_ratio
    return {field: bool(flags[index]) for index, field in enumerate(fields)}


def build_prefill(
    rng: np.random.Generator,
    sample_id: int,
    field_keep: dict[str, bool],
    gender_hint: str,
) -> dict:
    birth_date, birth_year = random_birth_date(rng)
    issue_year = int(rng.integers(birth_year + 14, 2026))
    issue_month = int(rng.integers(1, 13))
    issue_day = int(rng.integers(1, 29))
    work_years = int(rng.integers(0, 25))
    work_months = int(rng.integers(0, 12))
    prefill: dict = {}
    if field_keep["gender"]:
        prefill["gender"] = gender_hint
    if field_keep["birth_date"]:
        prefill["birth_date"] = birth_date
    if field_keep["birth_year"]:
        prefill["birth_year"] = birth_year
    if field_keep["passport_series_number"]:
        prefill["passport_series_number"] = random_passport_series_number(rng)
    if field_keep["passport_issue_date"]:
        prefill["passport_issue_date"] = f"{issue_day:02d}.{issue_month:02d}.{issue_year}"
    if field_keep["passport_department_code"]:
        prefill["passport_department_code"] = random_department_code(rng)
    if field_keep["inn"]:
        prefill["inn"] = random_inn(rng)
    if field_keep["snils"]:
        prefill["snils"] = random_snils(rng)
    if field_keep["mobile_phone"]:
        prefill["mobile_phone"] = random_phone(rng)
    if field_keep["email"]:
        prefill["email"] = random_email(rng, str(sample_id))
    if field_keep["monthly_income"]:
        prefill["monthly_income"] = str(int(rng.integers(35_000, 350_000)))
    if field_keep["dependents_count"]:
        prefill["dependents_count"] = int(rng.integers(0, 5))
    if field_keep["loans_count"]:
        prefill["loans_count"] = int(rng.integers(0, 4))
    if field_keep["work_experience"]:
        prefill["work_experience"] = WorkExperience(years=work_years, months=work_months)
    return prefill


def build_gold_spec(
    rng: np.random.Generator,
    sample_id: int,
    total: int,
    used_surnames: list[str],
    batch_slot: int,
) -> dict:
    keep_ratio = float(rng.uniform(0.2, 0.8))
    field_keep = field_mask(rng, GOLD_FIELDS, keep_ratio=keep_ratio)
    gender_hint = "м" if bool(rng.integers(0, 2)) else "ж"
    prefill = build_prefill(
        rng,
        sample_id=sample_id,
        field_keep=field_keep,
        gender_hint=gender_hint,
    )
    return {
        "sample_id": sample_id,
        "total": total,
        "batch_slot": batch_slot,
        "keep_ratio": keep_ratio,
        "field_mask": field_keep,
        "prefill": prefill,
        "used_surnames": list(used_surnames[-40:]),
        "diversity_key": int(rng.integers(0, 1_000_000_000)),
        "region_hint": str(rng.choice(REGION_HINTS)),
        "job_hint": str(rng.choice(JOB_HINTS)),
        "gender_hint": gender_hint,
    }
