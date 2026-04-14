from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"


def test_research_pipeline_smoke() -> None:
    with TestClient(app) as client:
        r = client.post("/research", json={"query": "What is Aurora compliance and what are the encryption requirements?"})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["original_query"]
        assert isinstance(data["subqueries"], list)
        assert isinstance(data["subquery_results"], list)
        assert data["final_answer"]
        assert "demonstration" in data
        assert data["demonstration"]["memory_pruning"]
        assert data["demonstration"]["episodic_compression"]
        assert data["demonstration"]["final_synthesis"]
        assert "session_metrics" in data
        assert data["session_metrics"]["discard_breakdown"] is not None
        assert "reviewer_note" in data
        assert isinstance(data["episodic_summaries"], list)
        assert "[Mock answer]" not in data["final_answer"]
        assert "[Mock answer]" not in "\n".join(
            r["subquery_answer"] for r in data["subquery_results"]
        )
        assert data["constraints"]["max_subqueries"] == 3
        sid = data["session_id"]
        g = client.get(f"/session/{sid}")
        assert g.status_code == 200
        body = g.json()
        assert body["session_id"] == sid
        assert body["subquery_results"]
