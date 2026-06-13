from __future__ import annotations

import re

from schemas.nerel import NerelEntity, NerelExtraction, NerelLink, NerelRelation
from schemas.runne import RunneEntity, RunneExtraction


def parse_nerel_entity_line(line: str, document_text: str) -> list[NerelEntity]:
    parts = line.split("\t")
    id_value = parts[0]
    type_and_spans = parts[1]
    entity_type, span_blob = type_and_spans.split(maxsplit=1)
    annotated_text = parts[2] if len(parts) > 2 else None
    if annotated_text == "":
        annotated_text = None
    entities = []
    span_groups = [group.strip() for group in span_blob.split(";")]
    for group in span_groups:
        start_str, end_str = group.split()
        start = int(start_str)
        end = int(end_str)
        if len(span_groups) == 1:
            text = annotated_text
        else:
            text = document_text[start:end]
        entities.append(
            NerelEntity(
                id=id_value,
                type=entity_type,
                start=start,
                end=end,
                text=text,
            )
        )
    return entities


def parse_nerel_relation_line(line: str) -> NerelRelation:
    parts = line.split("\t")
    id_value = parts[0]
    tail = parts[1]
    relation_type, arg1_raw, arg2_raw = tail.split()
    arg1 = arg1_raw.split(":", 1)[1]
    arg2 = arg2_raw.split(":", 1)[1]
    return NerelRelation(
        id=id_value,
        type=relation_type,
        arg1=arg1,
        arg2=arg2,
    )


def parse_nerel_link_line(line: str) -> NerelLink:
    parts = line.split("\t")
    id_value = parts[0]
    tail = parts[1].split()
    entity_id = tail[1]
    reference = tail[2]
    kb_name = parts[2] if len(parts) > 2 else None
    if kb_name == "":
        kb_name = None
    return NerelLink(
        id=id_value,
        entity_id=entity_id,
        reference=reference,
        kb_name=kb_name,
    )


def parse_nerel_gold(row: dict) -> NerelExtraction:
    document_text = row["text"]
    entities = []
    for line in row["entities"]:
        entities.extend(parse_nerel_entity_line(line, document_text=document_text))
    relations = [parse_nerel_relation_line(line) for line in row["relations"]]
    links = [parse_nerel_link_line(line) for line in row["links"]]
    return NerelExtraction(entities=entities, relations=relations, links=links)


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
