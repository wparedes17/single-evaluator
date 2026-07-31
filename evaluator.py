import os
from typing import Any, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


class EvaluationState(TypedDict):
    question: str
    answer: str
    criteria: str
    metadata: Optional[dict]
    score: int
    reason: str
    context_precision: Optional[float]
    context_recall: Optional[float]
    context_entities_recall: Optional[float]
    noise_sensitivity: Optional[float]
    response_relavancy: Optional[float]
    faithfulness: Optional[float]


class EvaluationOutput(BaseModel):
    score: int = Field(description="Likert scale score from 1 to 5", ge=1, le=5)
    reason: str = Field(description="Explanation for the score")
    context_precision: Optional[float] = Field(None, description="Proportion of retrieved context chunks that are relevant (0–1). Compute only when retrieval context is provided.")
    context_recall: Optional[float] = Field(None, description="How well the retrieved context covers the ground-truth answer (0–1). Compute only when retrieval context is provided.")
    context_entities_recall: Optional[float] = Field(None, description="Entity-level recall of the retrieved context (0–1). Compute only when retrieval context is provided.")
    noise_sensitivity: Optional[float] = Field(None, description="Sensitivity of the answer to irrelevant/noisy context chunks (0–1). Compute only when retrieval context is provided.")
    response_relavancy: Optional[float] = Field(None, description="How relevant the answer is to the question, irrespective of context (0–1). Compute only when retrieval context is provided.")
    faithfulness: Optional[float] = Field(None, description="How faithful the answer is to the retrieved context (0–1). Compute only when retrieval context is provided.")


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert evaluator. Evaluate the given answer to a question "
        "according to the provided criteria using the following Likert scale:\n"
        "1 - Very poor: answer fails to meet the criteria\n"
        "2 - Poor: answer partially meets the criteria with significant issues\n"
        "3 - Adequate: answer meets basic criteria but lacks depth or clarity\n"
        "4 - Good: answer meets the criteria well with only minor issues\n"
        "5 - Excellent: answer fully and excellently meets the criteria\n\n"
        "Return a score and a concise reason.\n\n"
        "When retrieval context is provided below, also compute each of the six "
        "RAG metrics (context_precision, context_recall, context_entities_recall, "
        "noise_sensitivity, response_relavancy, faithfulness) as a float in [0, 1]. "
        "Leave them null when no retrieval context is available."
    )),
    ("human", (
        "Question: {question}\n\n"
        "Answer: {answer}\n\n"
        "Criteria: {criteria}"
        "{metadata_section}"
    )),
])


def _format_metadata_section(metadata: Optional[dict]) -> str:
    if not metadata:
        return ""
    lines = ["\n\nRetrieval Context:"]
    intents = metadata.get("intents", [])
    if intents:
        lines.append(f"Intents: {', '.join(intents)}")
    for i, mod in enumerate(metadata.get("modules", []), 1):
        lines.append(
            f"\nModule {i} — {mod.get('module_name')} ({mod.get('module_type')}):\n"
            f"{mod.get('text_response', '')}"
        )
    return "\n".join(lines)


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
            "metadata_section": _format_metadata_section(state.get("metadata")),
        })
        return {
            "score": result.score,
            "reason": result.reason,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "context_entities_recall": result.context_entities_recall,
            "noise_sensitivity": result.noise_sensitivity,
            "response_relavancy": result.response_relavancy,
            "faithfulness": result.faithfulness,
        }

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
