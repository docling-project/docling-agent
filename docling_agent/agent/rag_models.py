"""Pydantic data models for the chunkless RAG loop."""

from pydantic import BaseModel


class SectionSelection(BaseModel):
    """LLM output: which section to consult next."""

    reason: str       # LLM's chain-of-thought for why this section is relevant
    section_ref: str  # self_ref of the selected section (e.g. "#/texts/3")


class AnswerAttempt(BaseModel):
    """LLM output: attempt to answer after reading a section."""

    can_answer: bool  # True if the LLM believes it has enough context
    response: str     # the answer (if can_answer) or what is still missing


class RAGIteration(BaseModel):
    """Record of a single RAG loop iteration (for logging / explainability)."""

    iteration: int
    section_ref: str
    reason: str
    section_text_length: int
    can_answer: bool
    response: str


class RAGResult(BaseModel):
    """Final result of a RAG run over a single document."""

    answer: str
    iterations: list[RAGIteration]
    converged: bool   # True if can_answer was reached; False if max_iterations hit
