from typing import Any, Optional

from pydantic import BaseModel, Field


class Module(BaseModel):
    module_name: str
    module_type: str
    text_response: str
    raw_input_data: dict[str, Any]
    raw_output_data: dict[str, Any]


class Metadata(BaseModel):
    intents: list[str]
    modules: list[Module]


class EvaluationRequest(BaseModel):
    question: str
    answer: str
    criteria: str
    metadata: Optional[Metadata] = None


class EvaluationResponse(BaseModel):
    score: int = Field(description="Likert scale score from 1 (very poor) to 5 (excellent)", ge=1, le=5)
    reason: str
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_entities_recall: Optional[float] = None
    noise_sensitivity: Optional[float] = None
    response_relavancy: Optional[float] = None  # intentional spelling per contract
    faithfulness: Optional[float] = None
