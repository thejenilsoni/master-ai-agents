from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ResearchReport


_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


@dataclass(frozen=True, slots=True)
class CitationAudit:
    valid: bool
    cited_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    uncited_source_ids: tuple[str, ...]
    paragraphs_without_citations: tuple[str, ...]


def audit_report_citations(report: ResearchReport) -> CitationAudit:
    text_parts = [report.executive_summary, *(section.body for section in report.sections)]
    text = "\n\n".join(text_parts)
    cited = tuple(dict.fromkeys(_CITATION_PATTERN.findall(text)))
    known = {source.source_id for source in report.sources}
    unknown = tuple(source_id for source_id in cited if source_id not in known)
    uncited = tuple(source_id for source_id in sorted(known) if source_id not in cited)

    paragraphs_without: list[str] = []
    for paragraph in (part.strip() for part in re.split(r"\n\s*\n", text)):
        if len(paragraph) < 80:
            continue
        has_factual_signal = bool(re.search(r"\b\d+(?:\.\d+)?%?\b|\b(according|reported|found|shows)\b", paragraph, re.I))
        if has_factual_signal and not _CITATION_PATTERN.search(paragraph):
            paragraphs_without.append(paragraph[:180])

    return CitationAudit(
        valid=not unknown and not paragraphs_without and bool(cited),
        cited_ids=cited,
        unknown_ids=unknown,
        uncited_source_ids=uncited,
        paragraphs_without_citations=tuple(paragraphs_without),
    )
