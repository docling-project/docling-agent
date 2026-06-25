"""Pydantic data models for the chunkless RAG loop."""

from typing import Annotated

from pydantic import BaseModel, Field


class SectionSelection(BaseModel):
    """LLM output: which section to consult next."""

    reason: Annotated[str, Field(description="LLM's chain-of-thought for why this section is relevant")]
    section_ref: Annotated[str, Field(description="self_ref of the selected section (e.g. '#/texts/3')")]


class AnswerAttempt(BaseModel):
    """LLM output: attempt to answer after reading a section."""

    can_answer: Annotated[bool, Field(description="True if the LLM believes it has enough context")]
    response: Annotated[str, Field(description="The answer (if can_answer) or what is still missing")]


class RAGIteration(BaseModel):
    """Record of a single RAG loop iteration (for logging / explainability)."""

    iteration: Annotated[int, Field(description="Iteration number in the RAG loop")]
    section_ref: Annotated[str, Field(description="Reference to the selected section")]
    reason: Annotated[str, Field(description="Reason for selecting this section")]
    section_text_length: Annotated[int, Field(description="Length of the section text in characters")]
    can_answer: Annotated[bool, Field(description="Whether the LLM can answer after this iteration")]
    response: Annotated[str, Field(description="The response or partial answer")]


class RAGResult(BaseModel):
    """Final result of a RAG run over a single document."""

    answer: Annotated[str, Field(description="The final answer to the query")]
    iterations: Annotated[list[RAGIteration], Field(description="List of all RAG iterations performed")]
    converged: Annotated[bool, Field(description="True if can_answer was reached; False if max_iterations hit")]


class RAGTrace(BaseModel):
    """Full reasoning trace of a RAG run across one or more documents."""

    query: Annotated[str, Field(description="The original query answered by this run")]
    per_document: Annotated[list[RAGResult], Field(description="One entry per source DoclingDocument, in input order")]
    final_answer: Annotated[str, Field(description="Merged answer that run() wraps into a DoclingDocument")]
