from __future__ import annotations

import re

from schemas.runne import RunneEntity, RunneExtraction


def parse_runne_entity_line(line: str) -> RunneEntity:
    start, end, entity_type = line.split()
    return RunneEntity(start=int(start), end=int(end), type=entity_type)


def parse_runne_gold(row: dict) -> RunneExtraction:
    entities = [parse_runne_entity_line(line) for line in row["entities"]]
    return RunneExtraction(entities=entities)


json_block_re = re.compile(r"\{.*\}", re.S)


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("{"):
        import json

        return json.loads(stripped)
    match = json_block_re.search(stripped)
    import json

    return json.loads(match.group(0))
