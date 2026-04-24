from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from evaluator import get_evaluator
from models import EvaluationRequest, EvaluationResponse

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_evaluator()
    yield


app = FastAPI(
    title="Answer Evaluator",
    description="Evaluates an answer for a given question and criteria using a 1-5 Likert scale.",
    lifespan=lifespan,
)


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    try:
        evaluator = get_evaluator()
        result = evaluator.invoke({
            "question": request.question,
            "answer": request.answer,
            "criteria": request.criteria,
            "score": 0,
            "reason": "",
        })
        return EvaluationResponse(score=result["score"], reason=result["reason"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
