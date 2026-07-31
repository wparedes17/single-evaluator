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


@app.post("/evaluate", response_model=EvaluationResponse, response_model_exclude_none=True)
async def evaluate(request: EvaluationRequest):
    try:
        evaluator = get_evaluator()
        result = evaluator.invoke({
            "question": request.question,
            "answer": request.answer,
            "criteria": request.criteria,
            "metadata": request.metadata.model_dump() if request.metadata else None,
            "score": 0,
            "reason": "",
            "context_precision": None,
            "context_recall": None,
            "context_entities_recall": None,
            "noise_sensitivity": None,
            "response_relavancy": None,
            "faithfulness": None,
        })
        return EvaluationResponse(
            score=result["score"],
            reason=result["reason"],
            context_precision=result.get("context_precision"),
            context_recall=result.get("context_recall"),
            context_entities_recall=result.get("context_entities_recall"),
            noise_sensitivity=result.get("noise_sensitivity"),
            response_relavancy=result.get("response_relavancy"),
            faithfulness=result.get("faithfulness"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
