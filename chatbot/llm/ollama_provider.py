"""Local Ollama LLM provider - default, zero-cost, offline."""
import requests

from chatbot.llm.base import LLMProvider
from chatbot.config import OLLAMA_HOST, OLLAMA_MODEL


class OllamaProvider(LLMProvider):
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, system_prompt: str, messages: list[dict], context: str) -> str:
        # Single combined system message, not two separate system-role turns -
        # keeps behavior consistent with OpenRouterProvider (some chat
        # templates only reliably honor one system message).
        full_messages = [
            {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"},
            *messages,
        ]
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": full_messages, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama unavailable: {e}") from e

        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
