from deep_research_agent.citations import audit_report_citations
from deep_research_agent.models import ReportSection, ResearchReport


def test_valid_report_citations(source_factory) -> None:
    report = ResearchReport(
        title="Report",
        executive_summary="The evidence supports this conclusion [S1].",
        sections=[ReportSection(heading="Analysis", body="A 2025 study reported a measurable effect [S1].")],
        key_findings=["Finding"],
        limitations=["Limited sample"],
        sources=[source_factory("S1", "https://example.com/report")],
        confidence=0.8,
    )
    audit = audit_report_citations(report)
    assert audit.valid is True
    assert audit.unknown_ids == ()


def test_unknown_citation_fails(source_factory) -> None:
    report = ResearchReport(
        title="Report",
        executive_summary="This claim relies on an unknown source [S9].",
        sections=[],
        key_findings=[],
        limitations=[],
        sources=[source_factory("S1", "https://example.com/report")],
        confidence=0.4,
    )
    audit = audit_report_citations(report)
    assert audit.valid is False
    assert audit.unknown_ids == ("S9",)
