from __future__ import annotations

import numpy as np


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
    return "".join(str(int(v)) for v in full)


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
    body = "".join(str(int(v)) for v in digits)
    return f"{body[:3]}-{body[3:6]}-{body[6:9]} {check:02d}"


def random_passport_series_number(rng: np.random.Generator) -> str:
    series = rng.integers(10, 99, size=2, dtype=np.int64)
    series_tail = rng.integers(0, 100, size=2, dtype=np.int64)
    number = rng.integers(0, 1_000_000, dtype=np.int64)
    return f"{series[0]:02d}{series[1]:02d} {series_tail[0]:02d}{series_tail[1]:02d} {number:06d}"


def random_department_code(rng: np.random.Generator) -> str:
    left = rng.integers(100, 1000, dtype=np.int64)
    right = rng.integers(0, 1000, dtype=np.int64)
    return f"{left:03d}-{right:03d}"


def random_birth_date(rng: np.random.Generator) -> tuple[str, int]:
    year = int(rng.integers(1955, 2006))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 29))
    return f"{day:02d}.{month:02d}.{year}", year


def random_phone(rng: np.random.Generator) -> str:
    tail = rng.integers(0, 10_000_000, dtype=np.int64)
    prefix = int(rng.choice(np.array([903, 905, 906, 909, 915, 916, 917, 919, 925, 926, 929, 977, 999], dtype=np.int64)))
    return f"+7{prefix}{tail:07d}"


def random_email(rng: np.random.Generator, token: str) -> str:
    domain = str(rng.choice(np.array(["mail.ru", "yandex.ru", "gmail.com", "inbox.ru"], dtype=object)))
    return f"client_{token}@{domain}"


def field_mask(rng: np.random.Generator, fields: list[str], fill_prob: float) -> dict[str, bool]:
    flags = rng.random(len(fields)) < fill_prob
    return {field: bool(flags[index]) for index, field in enumerate(fields)}


DETERMINISTIC_FIELDS = [
    "gender",
    "birth_date",
    "birth_year",
    "passport_series_number",
    "passport_issue_date",
    "passport_department_code",
    "inn",
    "snils",
    "mobile_phone",
    "email",
    "monthly_income",
    "dependents_count",
    "loans_count",
    "work_experience_years",
    "work_experience_months",
]


PERSON_FIELDS = [
    "birth_place",
    "citizenship",
    "passport_issued_by",
    "employer",
    "job_title",
    "marital_status",
    "real_estate",
    "car",
    "registration_address",
    "actual_address",
]


def build_skeleton(rng: np.random.Generator, sample_id: int) -> dict:
    det_mask = field_mask(rng, DETERMINISTIC_FIELDS, fill_prob=0.75)
    person_mask = field_mask(rng, PERSON_FIELDS, fill_prob=0.7)
    gender = "м" if bool(rng.integers(0, 2)) else "ж"
    birth_date, birth_year = random_birth_date(rng)
    issue_year = int(rng.integers(birth_year + 14, 2026))
    issue_month = int(rng.integers(1, 13))
    issue_day = int(rng.integers(1, 29))
    skeleton = {
        "sample_id": sample_id,
        "person_mask": person_mask,
        "det_mask": det_mask,
        "_gender": gender,
        "gender": gender if det_mask["gender"] else None,
        "birth_date": birth_date if det_mask["birth_date"] else None,
        "birth_year": birth_year if det_mask["birth_year"] else None,
        "passport_series_number": random_passport_series_number(rng) if det_mask["passport_series_number"] else None,
        "passport_issue_date": f"{issue_day:02d}.{issue_month:02d}.{issue_year}" if det_mask["passport_issue_date"] else None,
        "passport_department_code": random_department_code(rng) if det_mask["passport_department_code"] else None,
        "inn": random_inn(rng) if det_mask["inn"] else None,
        "snils": random_snils(rng) if det_mask["snils"] else None,
        "mobile_phone": random_phone(rng) if det_mask["mobile_phone"] else None,
        "email": random_email(rng, str(sample_id)) if det_mask["email"] else None,
        "monthly_income": str(int(rng.integers(35_000, 350_000))) if det_mask["monthly_income"] else None,
        "dependents_count": int(rng.integers(0, 5)) if det_mask["dependents_count"] else None,
        "loans_count": int(rng.integers(0, 4)) if det_mask["loans_count"] else None,
        "work_experience_years": int(rng.integers(0, 25)) if det_mask["work_experience_years"] else None,
        "work_experience_months": int(rng.integers(0, 12)) if det_mask["work_experience_months"] else None,
    }
    return skeleton
