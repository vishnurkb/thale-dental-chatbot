from fastapi.testclient import TestClient

import chatbot.api.main as main


class FakeProvider:
    def __init__(self, fn):
        self.fn = fn

    def generate(self, *a, **kw):
        return self.fn(*a, **kw)


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
    monkeypatch.setattr(main, "get_provider", lambda: FakeProvider(lambda *a, **kw: "Here is your answer."))

    client = TestClient(main.app)
    res = client.post("/chat", json={"session_id": "s2", "message": "how much is whitening?"})
    data = res.json()
    assert data["reply"] == "Here is your answer."
    assert "https://x/whitening" in data["sources"]


def test_chat_llm_unavailable_returns_fallback(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda msg: [])

    def boom(*a, **kw):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(main, "get_provider", lambda: FakeProvider(boom))
    client = TestClient(main.app)
    res = client.post("/chat", json={"session_id": "s3", "message": "hello"})
    assert "temporarily unavailable" in res.json()["reply"].lower()


def test_chat_missing_provider_returns_fallback(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda msg: [])
    monkeypatch.setattr(main, "get_provider", lambda: None)
    monkeypatch.setattr(main, "_llm_provider_error", "OPENROUTER_API_KEY is not set")

    client = TestClient(main.app)
    res = client.post("/chat", json={"session_id": "s4", "message": "hello"})
    assert "temporarily unavailable" in res.json()["reply"].lower()


def test_chat_retrieval_error_returns_fallback(monkeypatch):
    def boom(msg):
        raise Exception("collection does not exist")

    monkeypatch.setattr(main, "retrieve", boom)
    monkeypatch.setattr(main, "get_provider", lambda: FakeProvider(lambda *a, **kw: "unused"))

    client = TestClient(main.app)
    res = client.post("/chat", json={"session_id": "s5", "message": "hello"})
    assert "temporarily unavailable" in res.json()["reply"].lower()


def test_health():
    client = TestClient(main.app)
    res = client.get("/health")
    assert res.json() == {"status": "ok"}
