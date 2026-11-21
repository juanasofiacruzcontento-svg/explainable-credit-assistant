import pytest
from fastapi.testclient import TestClient

from credit_xai import monitoring, rag
from credit_xai.data import load_processed
from credit_xai.fairness import audit
from credit_xai.features import MODEL_FEATURES


@pytest.fixture(scope="module")
def client(project):
    from credit_xai.api import app

    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_explain_endpoint(client):
    application = client.get("/example", params={"rejected": True}).json()
    res = client.post("/explain", json=application).json()
    assert res["approved"] is False
    assert res["narration"]["text"]
    assert res["decision_id"]
    assert len(res["contributions"]) == len(MODEL_FEATURES)


def test_invalid_input_rejected(client):
    application = client.get("/example").json()
    application["savings"] = "moltissimi"
    assert client.post("/predict", json=application).status_code == 422
    application = client.get("/example").json()
    application["age"] = 5
    assert client.post("/predict", json=application).status_code == 422


def test_decisions_are_logged(client):
    before = len(monitoring.load_logged())
    client.post("/predict", json=client.get("/example").json())
    assert len(monitoring.load_logged()) == before + 1


def test_ask_endpoint(client):
    res = client.post("/ask", json={"question": "Ho diritto a una spiegazione della decisione?"})
    assert res.status_code == 200 and res.json()["sources"]


def test_rag_finds_right_article():
    titles = [c.title for c, _ in rag.get_retriever().retrieve("sorveglianza umana", k=2)]
    assert any("art. 14" in t for t in titles)


def test_fairness_audit(credit_model):
    res = audit(credit_model, load_processed("test"))
    assert set(res) == {"sex", "age_group", "foreign_worker"}
    for attr in res.values():
        assert 0 <= attr["demographic_parity_difference"] <= 1


def test_drift_detects_simulated_shift(project):
    train, test = load_processed("train"), load_processed("test")
    report = monitoring.drift_report(train, monitoring.simulate_drift(test))
    assert report.set_index("feature").loc["credit_amount", "stato"] == "DRIFT"
