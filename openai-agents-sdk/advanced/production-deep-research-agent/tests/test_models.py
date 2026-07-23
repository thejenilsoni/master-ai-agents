from deep_research_agent.models import ReportSection, ResearchReport


def test_report_markdown_includes_sources(source_factory) -> None:
    report = ResearchReport(
        title="Reliable Agents",
        executive_summary="Summary [S1].",
        sections=[ReportSection(heading="Architecture", body="Analysis [S1].")],
        key_findings=["Use evaluations"],
        limitations=["Evidence changes quickly"],
        sources=[source_factory("S1", "https://example.com")],
        confidence=0.8,
    )
    markdown = report.to_markdown()
    assert markdown.startswith("# Reliable Agents")
    assert "## Sources" in markdown
    assert "[S1]" in markdown
