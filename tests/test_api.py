from fastapi.testclient import TestClient

import chatbot.api.main as main


def test_chat_empty_message_returns_prompt():
    client = TestClient(main.app)
    res = client.post("/chat", json={"session_id": "s1", "message": "  "})
    assert res.status_code == 200
    assert "type a question" in res.json()["reply"].lower()


def test_chat_happy_path(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda msg: [
        {"metadata": {"page_type": "service", "title": "Whitening", "url": "https://x/whitening"},
         "text": "info", "score": 1.0}
    ])
    monkeypatch.setattr(main.llm_provider, "generate", lambda *a, **kw: "Here is your answer.")

    client = TestClient(main.app)
    res = client.post("/chat", json={"session_id": "s2", "message": "how much is whitening?"})
    data = res.json()
    assert data["reply"] == "Here is your answer."
    assert "https://x/whitening" in data["sources"]


def test_chat_ollama_unavailable_returns_fallback(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda msg: [])

    def boom(*a, **kw):
        raise RuntimeError("Ollama unavailable")

    monkeypatch.setattr(main.llm_provider, "generate", boom)
    client = TestClient(main.app)
    res = client.post("/chat", json={"session_id": "s3", "message": "hello"})
    assert "temporarily unavailable" in res.json()["reply"].lower()


def test_health():
    client = TestClient(main.app)
    res = client.get("/health")
    assert res.json() == {"status": "ok"}
