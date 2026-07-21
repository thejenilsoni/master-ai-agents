from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ResearchDepth(StrEnum):
    RAPID = "rapid"
    STANDARD = "standard"
    DEEP = "deep"


class SourceQuality(StrEnum):
    PRIMARY = "primary"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=10, max_length=2_000)]
    depth: ResearchDepth = ResearchDepth.STANDARD
    audience: str = "technical decision-makers"
    constraints: list[str] = Field(default_factory=list)
    required_domains: list[str] = Field(default_factory=list)
    excluded_domains: list[str] = Field(default_factory=list)
    recency_days: int | None = Field(default=None, ge=1, le=3_650)
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return " ".join(value.split())


class ResearchQuestion(BaseModel):
    id: str = Field(pattern=r"^Q\d+$")
    question: str = Field(min_length=10)
    rationale: str = Field(min_length=10)
    search_queries: list[str] = Field(min_length=1, max_length=6)
    required_evidence: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    objective: str
    scope: list[str]
    exclusions: list[str] = Field(default_factory=list)
    questions: list[ResearchQuestion] = Field(min_length=1, max_length=10)
    synthesis_criteria: list[str] = Field(default_factory=list)


class SourceRecord(BaseModel):
    source_id: str = Field(pattern=r"^S\d+$")
    title: str = Field(min_length=1)
    url: HttpUrl
    publisher: str = "Unknown"
    published_at: datetime | None = None
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    quality: SourceQuality = SourceQuality.MEDIUM
    is_primary: bool = False
    relevance_score: float = Field(default=0.5, ge=0, le=1)
    credibility_notes: str = ""


class EvidenceItem(BaseModel):
    evidence_id: str = Field(pattern=r"^E\d+$")
    source_id: str = Field(pattern=r"^S\d+$")
    question_id: str = Field(pattern=r"^Q\d+$")
    claim: str = Field(min_length=5)
    support: str = Field(min_length=5)
    direct_quote: str | None = Field(default=None, max_length=500)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ResearchFindings(BaseModel):
    question_id: str = Field(pattern=r"^Q\d+$")
    answer: str
    sources: list[SourceRecord] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    topic: str
    positions: list[str] = Field(min_length=2)
    source_ids: list[str] = Field(min_length=2)
    resolution: str
    confidence: float = Field(ge=0, le=1)


class Synthesis(BaseModel):
    key_findings: list[str]
    contradictions: list[Contradiction] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class ReportSection(BaseModel):
    heading: str
    body: str


class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    sections: list[ReportSection]
    key_findings: list[str]
    limitations: list[str]
    sources: list[SourceRecord]
    confidence: float = Field(ge=0, le=1)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", "## Executive Summary", "", self.executive_summary, ""]
        for section in self.sections:
            lines.extend([f"## {section.heading}", "", section.body, ""])
        lines.extend(["## Key Findings", ""])
        lines.extend(f"- {item}" for item in self.key_findings)
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in self.limitations)
        lines.extend(["", "## Sources", ""])
        for source in self.sources:
            lines.append(f"- [{source.source_id}] [{source.title}]({source.url}) — {source.publisher}")
        return "\n".join(lines).strip() + "\n"


class Critique(BaseModel):
    score: int = Field(ge=0, le=100)
    pass_threshold_met: bool
    strengths: list[str]
    factual_risks: list[str]
    citation_issues: list[str]
    missing_analysis: list[str]
    revision_instructions: list[str]


class ApprovalDecision(BaseModel):
    approved: bool
    reviewer: str = "human"
    notes: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
