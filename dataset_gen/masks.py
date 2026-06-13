from __future__ import annotations

import numpy as np


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


def field_mask(rng: np.random.Generator, fields: list[str], fill_prob: float) -> dict[str, bool]:
    flags = rng.random(len(fields)) < fill_prob
    return {field: bool(flags[index]) for index, field in enumerate(fields)}


def build_gold_spec(
    rng: np.random.Generator,
    sample_id: int,
    total: int,
    used_surnames: list[str],
) -> dict:
    field_keep = field_mask(rng, GOLD_FIELDS, fill_prob=0.72)
    field_keep["surname"] = True
    field_keep["name"] = True
    return {
        "sample_id": sample_id,
        "total": total,
        "field_mask": field_keep,
        "used_surnames": list(used_surnames[-30:]),
    }
