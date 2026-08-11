"""Cloud LLM provider via OpenRouter (OpenAI-compatible API). Stateless -
makes the backend deployable to serverless hosts (e.g. Vercel) since it no
longer depends on a locally-running model.

OpenRouter's free-tier models are a shared, uncapped-demand pool with no
uptime guarantee - any single one can return 429 (rate-limited) at any time.
generate() tries the primary model, then falls through a configured list of
fallback free models, so one congested model doesn't take the bot down."""
import time

import requests

from chatbot.llm.base import LLMProvider
from chatbot.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODELS

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model: str = OPENROUTER_MODEL,
        fallback_models: list[str] | None = None,
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.fallback_models = (
            fallback_models if fallback_models is not None else OPENROUTER_FALLBACK_MODELS
        )

    def _call(self, model: str, full_messages: list[dict]) -> requests.Response:
        return requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Thale Dental Chatbot",
            },
            json={"model": model, "messages": full_messages},
            timeout=60,
        )

    def generate(self, system_prompt: str, messages: list[dict], context: str) -> str:
        # Single combined system message, not two separate system-role turns:
        # some models' chat templates (e.g. gpt-oss's harmony format) only
        # reliably honor one system message and can otherwise miss the context.
        full_messages = [
            {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"},
            *messages,
        ]

        candidates = [self.model, *self.fallback_models]
        last_error = None

        for i, model in enumerate(candidates):
            for attempt in range(2):  # one retry per model, for transient errors
                try:
                    resp = self._call(model, full_messages)
                except requests.exceptions.RequestException as e:
                    last_error = f"{model}: {e}"
                    break  # network-level failure - move to next model, not worth retrying same one

                if resp.status_code == 429:
                    last_error = f"{model}: rate-limited (429)"
                    if attempt == 0:
                        time.sleep(1.5)
                        continue
                    break  # still rate-limited after retry - try next model

                if resp.status_code >= 400:
                    last_error = f"{model}: HTTP {resp.status_code} - {resp.text[:200]}"
                    break  # non-retryable error - move to next model

                try:
                    return resp.json()["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError, ValueError) as e:
                    last_error = f"{model}: unexpected response ({e})"
                    break

        raise RuntimeError(f"All OpenRouter models failed. Last error: {last_error}")
