from deep_research_agent.dedup import deduplicate_evidence, deduplicate_sources
from deep_research_agent.models import EvidenceItem


def test_sources_are_deduplicated_by_canonical_url(source_factory) -> None:
    sources = [
        source_factory("S1", "https://example.com/report?utm_source=x"),
        source_factory("S2", "https://EXAMPLE.com/report/#section"),
        source_factory("S3", "https://other.com/paper"),
    ]
    unique, remap = deduplicate_sources(sources)
    assert len(unique) == 2
    assert remap["S1"] == remap["S2"] == "S1"
    assert unique[1].source_id == "S2"


def test_evidence_is_remapped_and_near_duplicates_removed() -> None:
    items = [
        EvidenceItem(
            evidence_id="E1", source_id="S1", question_id="Q1", claim="Agent evaluation is necessary", support="Study A", confidence=0.8
        ),
        EvidenceItem(
            evidence_id="E2", source_id="S2", question_id="Q1", claim="Agent evaluation is necessary.", support="Study A duplicate", confidence=0.7
        ),
    ]
    unique = deduplicate_evidence(items, {"S1": "S1", "S2": "S1"}, similarity=0.85)
    assert len(unique) == 1
    assert unique[0].source_id == "S1"


def test_worker_local_source_ids_can_be_rekeyed_before_dedup(source_factory) -> None:
    sources = [
        source_factory("S1", "https://first.example/report"),
        source_factory("S2", "https://second.example/report"),
    ]
    unique, remap = deduplicate_sources(sources)
    assert [source.source_id for source in unique] == ["S1", "S2"]
    assert remap == {"S1": "S1", "S2": "S2"}
