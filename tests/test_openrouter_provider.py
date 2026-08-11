import requests

from chatbot.llm.openrouter_provider import OpenRouterProvider


class FakeResponse:
    def __init__(self, json_data, status=200, text=""):
        self._json = json_data
        self.status_code = status
        self.text = text or str(json_data)

    def json(self):
        return self._json


def test_generate_returns_message_content(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse({"choices": [{"message": {"content": "Hello from cloud"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OpenRouterProvider(api_key="fake-key", model="test-model", fallback_models=[])
    result = provider.generate("system", [{"role": "user", "content": "hi"}], "context")
    assert result == "Hello from cloud"


def test_falls_back_to_next_model_on_rate_limit(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["model"])
        if json["model"] == "primary":
            return FakeResponse({}, status=429)
        return FakeResponse({"choices": [{"message": {"content": "from fallback"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("chatbot.llm.openrouter_provider.time.sleep", lambda s: None)
    provider = OpenRouterProvider(api_key="fake-key", model="primary", fallback_models=["backup"])
    result = provider.generate("system", [], "context")
    assert result == "from fallback"
    assert "primary" in calls and "backup" in calls


def test_raises_runtime_error_when_all_models_fail(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse({}, status=429)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("chatbot.llm.openrouter_provider.time.sleep", lambda s: None)
    provider = OpenRouterProvider(api_key="fake-key", model="primary", fallback_models=["backup"])
    raised = False
    try:
        provider.generate("system", [], "context")
    except RuntimeError:
        raised = True
    assert raised


def test_generate_raises_runtime_error_when_unreachable(monkeypatch):
    def fake_post(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OpenRouterProvider(api_key="fake-key", fallback_models=[])
    raised = False
    try:
        provider.generate("system", [], "context")
    except RuntimeError:
        raised = True
    assert raised


def test_missing_api_key_raises():
    raised = False
    try:
        OpenRouterProvider(api_key="")
    except RuntimeError:
        raised = True
    assert raised
