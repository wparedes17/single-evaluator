import os
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


class EvaluationState(TypedDict):
    question: str
    answer: str
    criteria: str
    score: int
    reason: str


class EvaluationOutput(BaseModel):
    score: int = Field(description="Likert scale score from 1 to 5", ge=1, le=5)
    reason: str = Field(description="Explanation for the score")


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert evaluator. Evaluate the given answer to a question "
        "according to the provided criteria using the following Likert scale:\n"
        "1 - Very poor: answer fails to meet the criteria\n"
        "2 - Poor: answer partially meets the criteria with significant issues\n"
        "3 - Adequate: answer meets basic criteria but lacks depth or clarity\n"
        "4 - Good: answer meets the criteria well with only minor issues\n"
        "5 - Excellent: answer fully and excellently meets the criteria\n\n"
        "Return a score and a concise reason for your evaluation."
    )),
    ("human", (
        "Question: {question}\n\n"
        "Answer: {answer}\n\n"
        "Criteria: {criteria}"
    )),
])

_graph = None


def _build_graph():
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    structured_llm = llm.with_structured_output(EvaluationOutput)
    chain = _PROMPT | structured_llm

    def evaluate_node(state: EvaluationState) -> EvaluationState:
        result: EvaluationOutput = chain.invoke({
            "question": state["question"],
            "answer": state["answer"],
            "criteria": state["criteria"],
        })
        return {"score": result.score, "reason": result.reason}

    graph = StateGraph(EvaluationState)
    graph.add_node("evaluate", evaluate_node)
    graph.set_entry_point("evaluate")
    graph.add_edge("evaluate", END)
    return graph.compile()


def get_evaluator():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph
