from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, StringConstraints

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class PlanFields(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class RecommendPlan(PlanFields):
    action: Literal["recommend"]
    genres: list[ShortText] = Field(max_length=5)
    genre_mode: Literal["and", "or"]
    count: int = Field(ge=1, le=10)
    exclude_titles: list[ShortText] = Field(max_length=10)


class LookupPlan(PlanFields):
    action: Literal["lookup"]
    title: ShortText


class MessagePlan(PlanFields):
    action: Literal["clarify", "unsupported"]
    message: ShortText


Plan = RootModel[RecommendPlan | LookupPlan | MessagePlan]
