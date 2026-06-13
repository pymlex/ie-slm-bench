from __future__ import annotations

import json
import re

from pydantic import BaseModel


json_block_re = re.compile(r"\{.*\}", re.S)


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    match = json_block_re.search(stripped)
    return json.loads(match.group(0))


def load_extraction(raw_json: str, model_cls: type[BaseModel]) -> BaseModel:
    payload = json.loads(raw_json)
    return model_cls.model_validate(payload)


def parse_outlines_output(raw: str | BaseModel | dict, model_cls: type[BaseModel]) -> BaseModel:
    if isinstance(raw, model_cls):
        return raw
    if isinstance(raw, dict):
        return model_cls.model_validate(raw)
    return model_cls.model_validate_json(str(raw))
