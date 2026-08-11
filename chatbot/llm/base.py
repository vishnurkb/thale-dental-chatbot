"""LLM provider interface. Any provider (local Ollama, future cloud API) implements
this so the backend never depends on a specific provider's SDK."""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, messages: list[dict], context: str) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}, ...]
        Returns the assistant's reply text. Raises RuntimeError if the provider
        is unavailable."""
        raise NotImplementedError
