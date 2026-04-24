from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    question: str
    answer: str
    criteria: str


class EvaluationResponse(BaseModel):
    score: int = Field(description="Likert scale score from 1 (very poor) to 5 (excellent)", ge=1, le=5)
    reason: str = Field(description="Explanation for the score")
