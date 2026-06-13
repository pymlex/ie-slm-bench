from pydantic import BaseModel, Field


class RunneEntity(BaseModel):
    start: int
    end: int
    type: str


class RunneExtraction(BaseModel):
    entities: list[RunneEntity] = Field(default_factory=list)
