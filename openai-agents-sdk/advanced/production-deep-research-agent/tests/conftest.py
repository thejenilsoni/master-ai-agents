from __future__ import annotations

from datetime import UTC, datetime

import pytest

from deep_research_agent.models import SourceQuality, SourceRecord


@pytest.fixture
def source_factory():
    def create(source_id: str, url: str, title: str = "Source") -> SourceRecord:
        return SourceRecord(
            source_id=source_id,
            title=title,
            url=url,
            publisher="Example",
            accessed_at=datetime.now(UTC),
            quality=SourceQuality.HIGH,
            relevance_score=0.9,
        )

    return create
