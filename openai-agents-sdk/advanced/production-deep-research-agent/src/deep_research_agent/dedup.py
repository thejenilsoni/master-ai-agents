from __future__ import annotations

from difflib import SequenceMatcher

from .models import EvidenceItem, SourceRecord
from .security import canonicalize_url


def deduplicate_sources(sources: list[SourceRecord]) -> tuple[list[SourceRecord], dict[str, str]]:
    unique: list[SourceRecord] = []
    canonical_to_id: dict[str, str] = {}
    remap: dict[str, str] = {}

    for source in sources:
        canonical = canonicalize_url(str(source.url))
        existing_id = canonical_to_id.get(canonical)
        if existing_id:
            remap[source.source_id] = existing_id
            continue
        new_id = f"S{len(unique) + 1}"
        remap[source.source_id] = new_id
        canonical_to_id[canonical] = new_id
        unique.append(SourceRecord.model_validate({**source.model_dump(), "source_id": new_id, "url": canonical}))
    return unique, remap


def deduplicate_evidence(
    evidence: list[EvidenceItem], source_remap: dict[str, str], similarity: float = 0.9
) -> list[EvidenceItem]:
    unique: list[EvidenceItem] = []
    for item in evidence:
        remapped = item.model_copy(
            update={"source_id": source_remap.get(item.source_id, item.source_id)}
        )
        if any(
            SequenceMatcher(None, remapped.claim.lower(), existing.claim.lower()).ratio() >= similarity
            and remapped.source_id == existing.source_id
            for existing in unique
        ):
            continue
        unique.append(remapped.model_copy(update={"evidence_id": f"E{len(unique) + 1}"}))
    return unique
