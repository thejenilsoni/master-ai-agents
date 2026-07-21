from deep_research_agent.models import ApprovalDecision, ResearchRequest
from deep_research_agent.storage import ResearchStore


def test_store_lifecycle(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    request = ResearchRequest(query="What makes production agent systems reliable in practice?")
    store.create_run(request)
    row = store.load(request.run_id)
    assert row is not None
    assert row["status"] == "created"
    store.save_approval(request.run_id, ApprovalDecision(approved=True, reviewer="test"))
    assert store.load(request.run_id)["status"] == "approved"
