import requests

from chatbot.llm.ollama_provider import OllamaProvider


class FakeResponse:
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError("error")

    def json(self):
        return self._json


def test_generate_returns_message_content(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse({"message": {"content": "Hello from bot"}})

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaProvider(host="http://fake:11434", model="test-model")
    result = provider.generate("system", [{"role": "user", "content": "hi"}], "context")
    assert result == "Hello from bot"


def test_generate_raises_runtime_error_when_unreachable(monkeypatch):
    def fake_post(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaProvider()
    raised = False
    try:
        provider.generate("system", [], "context")
    except RuntimeError:
        raised = True
    assert raised
