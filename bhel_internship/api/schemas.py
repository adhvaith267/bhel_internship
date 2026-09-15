"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user question to answer")
    top_n: Optional[int] = Field(
        default=5, ge=1, le=20, description="Number of context chunks to use"
    )
    doc_name: Optional[str] = Field(
        default=None, description="Optional document name to filter context by"
    )


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_n: Optional[int] = Field(default=5, ge=1, le=20)
    stream: Optional[bool] = Field(default=False)
    doc_name: Optional[str] = Field(default=None)


class SourceCitation(BaseModel):
    document: str
    page: int
    score: float
    snippet: str
    is_table: Optional[bool] = False


class UnverifiedCitation(BaseModel):
    document: str
    page: int


class GroundingReport(BaseModel):
    cited_count: int = 0
    verified_count: int = 0
    grounding_rate: float = 1.0
    verified: List[UnverifiedCitation] = Field(default_factory=list)
    unverified: List[UnverifiedCitation] = Field(default_factory=list)


class AskResponse(BaseModel):
    response: str
    raw_text: str
    sources: List[SourceCitation]
    latency_ms: float
    model: str
    abstained: bool = False
    grounding: GroundingReport = Field(default_factory=GroundingReport)
