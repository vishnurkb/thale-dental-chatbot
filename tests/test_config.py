import importlib


def test_config_static_defaults():
    """Values with no equivalent key in the repo's real .env - these can only
    ever come from the hardcoded Python default, so they're a stable check
    regardless of what LLM_PROVIDER/etc. the local .env is actually set to."""
    import chatbot.config as config
    importlib.reload(config)
    assert "faq" in config.EVERGREEN_TYPES
    assert "contact" in config.EVERGREEN_TYPES


def test_config_respects_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("EVERGREEN_BOOST", "0.5")
    import chatbot.config as config
    importlib.reload(config)
    assert config.LLM_PROVIDER == "ollama"
    assert config.TOP_K == 9
    assert config.EVERGREEN_BOOST == 0.5
