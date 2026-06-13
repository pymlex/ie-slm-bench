from pydantic import BaseModel, Field


class NerelEntity(BaseModel):
    id: str
    type: str
    start: int
    end: int
    text: str | None = None


class NerelRelation(BaseModel):
    id: str
    type: str
    arg1: str
    arg2: str


class NerelLink(BaseModel):
    id: str
    entity_id: str
    reference: str
    kb_name: str | None = None


class NerelExtraction(BaseModel):
    entities: list[NerelEntity] = Field(default_factory=list)
    relations: list[NerelRelation] = Field(default_factory=list)
    links: list[NerelLink] = Field(default_factory=list)
