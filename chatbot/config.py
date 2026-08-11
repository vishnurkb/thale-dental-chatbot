import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SITE_DIR = Path(os.getenv("SITE_DIR", str(BASE_DIR / "site")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chatbot" / "index_store")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "thale_dental")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "300"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))
TOP_K = int(os.getenv("TOP_K", "8"))
EVERGREEN_BOOST = float(os.getenv("EVERGREEN_BOOST", "0.08"))
EVERGREEN_TYPES = {"faq", "service", "fee", "doctor", "contact"}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
# Tried in order if the primary is rate-limited/unavailable - OpenRouter's free
# tier is a shared pool with no uptime guarantee, so a single free model is not
# reliable enough on its own for a deployed backend.
OPENROUTER_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_FALLBACK_MODELS",
        "openai/gpt-oss-20b:free,nvidia/nemotron-nano-9b-v2:free,openrouter/free",
    ).split(",")
    if m.strip()
]

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))
