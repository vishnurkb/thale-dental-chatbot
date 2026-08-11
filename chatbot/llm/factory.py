"""Picks the active LLMProvider based on config.LLM_PROVIDER - the only place
that needs to know about concrete provider classes."""
from chatbot.config import LLM_PROVIDER
from chatbot.llm.base import LLMProvider


def get_llm_provider() -> LLMProvider:
    if LLM_PROVIDER == "openrouter":
        from chatbot.llm.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider()
    if LLM_PROVIDER == "ollama":
        from chatbot.llm.ollama_provider import OllamaProvider
        return OllamaProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
