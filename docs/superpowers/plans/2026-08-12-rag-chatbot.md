# Thale Dental RAG Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a free, fully local RAG chatbot that answers visitor questions using content from the `site/` mirror (109 pages), served via a FastAPI backend and a browser chat widget.

**Architecture:** Ingestion lane (parse HTML → structure-aware chunks → local embeddings → ChromaDB) runs offline. Query lane (chat widget → FastAPI `/chat` → retriever with evergreen-boosting → Ollama LLM via a swappable provider interface → grounded answer) runs at request time. See `docs/superpowers/specs/2026-08-12-rag-chatbot-design.md` for full design.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, BeautifulSoup4, sentence-transformers (`all-MiniLM-L6-v2`), ChromaDB, Ollama (local LLM), pytest, vanilla HTML/CSS/JS widget, Docker.

## Note on commit steps

This directory has no git repo yet (`git init` never run). Commit steps below are
written per plan convention but are optional here — skip them, or run `git init`
once up front, if you want history. Not required for the chatbot to work.

## Global Constraints

- Zero cost: no paid API keys, no paid hosting.
- Runs fully offline/local: Ollama for LLM, sentence-transformers for embeddings, Chroma persisted to local disk.
- No booking/PMS integration — Q&A only, per spec.
- `LLMProvider` must be an interface (`chatbot/llm/base.py`) so a future cloud provider can be added without touching backend logic.
- All tunables in `chatbot/config.py`, sourced from env vars with sensible defaults.
- Fee-related answers must carry the disclaimer defined in the spec.
- Bot must never fabricate an answer outside retrieved context (system prompt enforces this).

---

## File Structure

```
chatbot/
  __init__.py
  config.py
  ingest/
    __init__.py
    parser.py
    build_index.py
  retrieval/
    __init__.py
    retriever.py
  llm/
    __init__.py
    base.py
    ollama_provider.py
  prompt.py
  api/
    __init__.py
    main.py
  widget/
    widget.html
tests/
  test_config.py
  test_parser.py
  test_build_index.py
  test_retriever.py
  test_ollama_provider.py
  test_prompt.py
  test_api.py
requirements.txt
.env.example
Dockerfile
```

---

### Task 1: Project scaffold + config

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `chatbot/__init__.py` (empty)
- Create: `chatbot/config.py`
- Test: `tests/test_config.py`
- Create: `tests/__init__.py` (empty)

**Interfaces:**
- Produces: `chatbot.config` module exposing `SITE_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHUNK_SIZE_TOKENS, TOP_K, EVERGREEN_BOOST, EVERGREEN_TYPES, LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_HOST, API_HOST, API_PORT` (all module-level constants).

- [ ] **Step 1: Create requirements.txt**

```
fastapi
uvicorn[standard]
beautifulsoup4
sentence-transformers
chromadb
requests
python-dotenv
pydantic
pytest
httpx
```

- [ ] **Step 2: Create .env.example**

```
SITE_DIR=./site
CHROMA_DIR=./chatbot/index_store
COLLECTION_NAME=thale_dental
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE_TOKENS=300
CHUNK_OVERLAP_TOKENS=40
TOP_K=5
EVERGREEN_BOOST=0.08
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
API_HOST=0.0.0.0
API_PORT=8001
```

- [ ] **Step 3: Write the failing test**

`tests/test_config.py`:
```python
import importlib


def test_config_defaults(monkeypatch):
    for var in ["TOP_K", "LLM_PROVIDER", "EVERGREEN_BOOST"]:
        monkeypatch.delenv(var, raising=False)
    import chatbot.config as config
    importlib.reload(config)
    assert config.TOP_K == 5
    assert config.LLM_PROVIDER == "ollama"
    assert "faq" in config.EVERGREEN_TYPES
    assert "contact" in config.EVERGREEN_TYPES
    assert config.EVERGREEN_BOOST == 0.08
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chatbot'`

- [ ] **Step 5: Create chatbot/config.py**

```python
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
TOP_K = int(os.getenv("TOP_K", "5"))
EVERGREEN_BOOST = float(os.getenv("EVERGREEN_BOOST", "0.08"))
EVERGREEN_TYPES = {"faq", "service", "fee", "doctor", "contact"}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example chatbot/__init__.py chatbot/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: project scaffold and env-driven config"
```

---

### Task 2: Structure-aware HTML parser

**Files:**
- Create: `chatbot/ingest/__init__.py` (empty)
- Create: `chatbot/ingest/parser.py`
- Test: `tests/test_parser.py`

**Interfaces:**
- Consumes: nothing (pure parsing module, stdlib + bs4 only)
- Produces: `Chunk` dataclass `{text: str, page_type: str, title: str, url: str, chunk_id: str}`; `parse_page(html_path: Path, rel_path: str, base_url: str) -> list[Chunk]`; `classify_page_type(rel_path: str) -> str`. Consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

`tests/test_parser.py`:
```python
from chatbot.ingest.parser import parse_page

FAQ_HTML = """
<html><head><title>FAQ</title></head><body>
<main>
<h2>Do you accept new patients?</h2>
<p>Yes, we are currently accepting new patients.</p>
<h2>Do you offer emergency appointments?</h2>
<p>Yes, same-day emergency slots are available.</p>
</main>
</body></html>
"""

FEE_HTML = """
<html><head><title>Our Fees</title></head><body>
<main>
<table>
<tr><th>Treatment</th><th>Price</th></tr>
<tr><td>Check-up</td><td>&pound;50</td></tr>
<tr><td>Teeth Whitening</td><td>&pound;350</td></tr>
</table>
</main>
</body></html>
"""

DOCTOR_HTML = """
<html><head><title>Dr Jasdeep Singh Bahra</title></head><body>
<main><p>Dr Bahra is our principal dentist with 15 years experience.</p></main>
</body></html>
"""

GENERIC_HTML = """
<html><head><title>About</title></head><body>
<main><p>Theale Dental is a friendly practice in Theale offering general and cosmetic dentistry.</p></main>
</body></html>
"""


def write(tmp_path, rel, content):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_faq_splits_per_question(tmp_path):
    path = write(tmp_path, "faq/index.html", FAQ_HTML)
    chunks = parse_page(path, "faq/index.html", "https://thealedental.co.uk")
    assert len(chunks) == 2
    assert chunks[0].page_type == "faq"
    assert "accepting new patients" in chunks[0].text


def test_parse_fees_keeps_price_with_treatment(tmp_path):
    path = write(tmp_path, "our-fees/index.html", FEE_HTML)
    chunks = parse_page(path, "our-fees/index.html", "https://thealedental.co.uk")
    assert any("Teeth Whitening" in c.text and "350" in c.text for c in chunks)
    assert all(c.page_type == "fee" for c in chunks)


def test_parse_doctor_single_chunk_tagged(tmp_path):
    path = write(tmp_path, "doctor/dr-jasdeep-singh-bahra/index.html", DOCTOR_HTML)
    chunks = parse_page(path, "doctor/dr-jasdeep-singh-bahra/index.html", "https://thealedental.co.uk")
    assert len(chunks) == 1
    assert chunks[0].page_type == "doctor"
    assert "Bahra" in chunks[0].title


def test_parse_generic_falls_back_to_blog_type(tmp_path):
    path = write(tmp_path, "some-post/index.html", GENERIC_HTML)
    chunks = parse_page(path, "some-post/index.html", "https://thealedental.co.uk")
    assert len(chunks) == 1
    assert chunks[0].page_type == "blog"
    assert "cosmetic dentistry" in chunks[0].text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chatbot.ingest'`

- [ ] **Step 3: Implement chatbot/ingest/parser.py**

```python
"""Parse mirrored HTML pages into (text, metadata) chunks, structure-aware per page type."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

from bs4 import BeautifulSoup


@dataclass
class Chunk:
    text: str
    page_type: str
    title: str
    url: str
    chunk_id: str


def classify_page_type(rel_path: str) -> str:
    p = rel_path.replace("\\", "/").lower()
    if p.startswith("faq/"):
        return "faq"
    if p.startswith("our-fees/"):
        return "fee"
    if p.startswith("doctor/"):
        return "doctor"
    if p.startswith("contact/"):
        return "contact"
    if p.startswith(("service/", "treatment/", "careplan/", "denplan/",
                      "childrens-check-ups/", "testimonials/")):
        return "service"
    return "blog"


def _clean_text(el) -> str:
    text = el.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return _clean_text(h1) if h1 else "Untitled"


def _main_content(soup: BeautifulSoup):
    for sel in ["main", "article", "#content", ".entry-content", "body"]:
        node = soup.select_one(sel)
        if node:
            return node
    return soup


def parse_faq(soup: BeautifulSoup, title: str, url: str) -> list[Chunk]:
    content = _main_content(soup)
    chunks = []
    headers = content.find_all(["h2", "h3"])
    for i, h in enumerate(headers):
        question = _clean_text(h)
        if not question:
            continue
        parts = []
        for sib in h.find_next_siblings():
            if sib.name in ("h2", "h3"):
                break
            txt = _clean_text(sib)
            if txt:
                parts.append(txt)
        answer = " ".join(parts).strip()
        if not answer:
            continue
        chunks.append(Chunk(
            text=f"Q: {question}\nA: {answer}",
            page_type="faq",
            title=question,
            url=url,
            chunk_id=f"{url}#faq-{i}",
        ))
    return chunks


def parse_fees(soup: BeautifulSoup, title: str, url: str) -> list[Chunk]:
    content = _main_content(soup)
    chunks = []
    rows = content.find_all("tr")
    for i, row in enumerate(rows):
        cells = [_clean_text(c) for c in row.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        treatment, price = cells[0], cells[-1]
        chunks.append(Chunk(
            text=f"{treatment}: {price}",
            page_type="fee",
            title=treatment,
            url=url,
            chunk_id=f"{url}#fee-{i}",
        ))
    if not chunks:
        text = _clean_text(content)
        if text:
            chunks.append(Chunk(text=text, page_type="fee", title=title, url=url, chunk_id=f"{url}#fee-0"))
    return chunks


def parse_doctor(soup: BeautifulSoup, title: str, url: str) -> list[Chunk]:
    content = _main_content(soup)
    text = _clean_text(content)
    if not text:
        return []
    return [Chunk(text=f"{title}: {text}", page_type="doctor", title=title, url=url, chunk_id=f"{url}#doctor-0")]


def parse_contact(soup: BeautifulSoup, title: str, url: str) -> list[Chunk]:
    content = _main_content(soup)
    text = _clean_text(content)
    if not text:
        return []
    return [Chunk(text=text, page_type="contact", title=title, url=url, chunk_id=f"{url}#contact-0")]


def chunk_generic_text(text: str, page_type: str, title: str, url: str,
                        chunk_size_words: int = 220, overlap_words: int = 30) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        piece = " ".join(words[start:end])
        chunks.append(Chunk(text=piece, page_type=page_type, title=title, url=url, chunk_id=f"{url}#chunk-{idx}"))
        idx += 1
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def parse_page(html_path: Path, rel_path: str, base_url: str) -> list[Chunk]:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    title = _page_title(soup)

    clean_rel = rel_path.replace("\\", "/").removesuffix("index.html").rstrip("/")
    url = f"{base_url.rstrip('/')}/{clean_rel}".rstrip("/") + "/"

    page_type = classify_page_type(rel_path)

    if page_type == "faq":
        return parse_faq(soup, title, url)
    if page_type == "fee":
        return parse_fees(soup, title, url)
    if page_type == "doctor":
        return parse_doctor(soup, title, url)
    if page_type == "contact":
        return parse_contact(soup, title, url)

    content = _main_content(soup)
    text = _clean_text(content)
    return chunk_generic_text(text, page_type, title, url)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add chatbot/ingest/__init__.py chatbot/ingest/parser.py tests/test_parser.py
git commit -m "feat: structure-aware HTML parser (faq/fee/doctor/contact/generic)"
```

---

### Task 3: Embedding + vector store ingestion

**Files:**
- Create: `chatbot/ingest/build_index.py`
- Test: `tests/test_build_index.py`

**Interfaces:**
- Consumes: `chatbot.ingest.parser.parse_page`, `chatbot.config.{SITE_DIR,CHROMA_DIR,COLLECTION_NAME,EMBEDDING_MODEL}`
- Produces: `build_index(site_dir: Path = SITE_DIR, chroma_dir: Path = CHROMA_DIR, base_url: str = "https://thealedental.co.uk") -> int` (returns chunk count). Writes a persistent Chroma collection named `COLLECTION_NAME` at `chroma_dir`. Consumed by Task 4 (retriever reads the same collection).

- [ ] **Step 1: Write the failing test**

`tests/test_build_index.py`:
```python
import chromadb

from chatbot.ingest.build_index import build_index

SIMPLE_HTML = """
<html><head><title>About</title></head><body>
<main><p>Theale Dental is a friendly practice in Theale offering general and cosmetic dentistry.</p></main>
</body></html>
"""


def test_build_index_creates_queryable_collection(tmp_path):
    site_dir = tmp_path / "site"
    (site_dir / "about").mkdir(parents=True)
    (site_dir / "about" / "index.html").write_text(SIMPLE_HTML, encoding="utf-8")

    chroma_dir = tmp_path / "index_store"
    count = build_index(site_dir=site_dir, chroma_dir=chroma_dir)

    assert count == 1

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection("thale_dental")
    assert collection.count() == count
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chatbot.ingest.build_index'`

- [ ] **Step 3: Implement chatbot/ingest/build_index.py**

```python
"""Build (or rebuild) the Chroma vector index from a site mirror. Run as a script
whenever site content changes; the API backend only ever reads this index."""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chatbot.config import SITE_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from chatbot.ingest.parser import parse_page


def iter_html_files(site_dir: Path):
    for path in sorted(Path(site_dir).rglob("*.html")):
        rel = path.relative_to(site_dir)
        yield path, str(rel)


def build_index(site_dir: Path = SITE_DIR, chroma_dir: Path = CHROMA_DIR,
                 base_url: str = "https://thealedental.co.uk") -> int:
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_chunks = []
    for path, rel in iter_html_files(site_dir):
        try:
            chunks = parse_page(path, rel, base_url)
        except Exception as e:
            print(f"[skip] {rel}: {e}")
            continue
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced.")
        return 0

    texts = [c.text for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=[c.chunk_id for c in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"page_type": c.page_type, "title": c.title, "url": c.url} for c in all_chunks],
    )
    print(f"Indexed {len(all_chunks)} chunks.")
    return len(all_chunks)


if __name__ == "__main__":
    build_index()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_index.py -v`
Expected: PASS (first run downloads `all-MiniLM-L6-v2`, ~80MB, one-time, free)

- [ ] **Step 5: Commit**

```bash
git add chatbot/ingest/build_index.py tests/test_build_index.py
git commit -m "feat: ingestion pipeline - embed chunks into persistent Chroma index"
```

---

### Task 4: Retriever with evergreen boosting + contact injection

**Files:**
- Create: `chatbot/retrieval/__init__.py` (empty)
- Create: `chatbot/retrieval/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `chatbot.config.{CHROMA_DIR,COLLECTION_NAME,EMBEDDING_MODEL,TOP_K,EVERGREEN_BOOST,EVERGREEN_TYPES}`
- Produces: `retrieve(query: str, top_k: int = TOP_K, collection=None, model=None) -> list[dict]` where each dict is `{id, text, metadata: {page_type,title,url}, score}`. Consumed by Task 7 (API backend).

- [ ] **Step 1: Write the failing tests**

`tests/test_retriever.py`:
```python
import uuid

import chromadb

from chatbot.retrieval.retriever import retrieve


class FakeModel:
    """Returns identical embeddings for every input so distance is always 0 and
    only the evergreen-boost logic decides ranking - isolates the boost behavior
    under test from real embedding quality."""
    def encode(self, texts):
        return [[1.0] for _ in texts]


def make_collection():
    client = chromadb.Client()  # ephemeral in-memory, no disk writes
    # unique name per call: chromadb.Client() shares its in-memory backend across
    # calls within the same process, so a fixed name collides across tests.
    collection = client.create_collection(f"test-{uuid.uuid4().hex}")
    collection.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0], [1.0], [1.0]],
        documents=[
            "old blog post about a staff wedding",
            "teeth whitening costs 350",
            "practice contact info and opening hours",
        ],
        metadatas=[
            {"page_type": "blog", "title": "Wedding", "url": "https://x/wedding"},
            {"page_type": "fee", "title": "Whitening", "url": "https://x/fees"},
            {"page_type": "contact", "title": "Contact", "url": "https://x/contact"},
        ],
    )
    return collection


def test_evergreen_types_ranked_above_blog_on_tie():
    collection = make_collection()
    hits = retrieve("teeth whitening cost", top_k=2, collection=collection, model=FakeModel())
    types = [h["metadata"]["page_type"] for h in hits]
    assert "blog" not in types


def test_contact_chunk_always_included():
    collection = make_collection()
    hits = retrieve("teeth whitening cost", top_k=1, collection=collection, model=FakeModel())
    assert any(h["metadata"]["page_type"] == "contact" for h in hits)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chatbot.retrieval'`

- [ ] **Step 3: Implement chatbot/retrieval/retriever.py**

```python
"""Query-time retrieval: embed the question, similarity search, boost evergreen
page types over stale blog content, always surface the contact info chunk."""
import chromadb
from sentence_transformers import SentenceTransformer

from chatbot.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, EVERGREEN_BOOST, EVERGREEN_TYPES

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, top_k: int = TOP_K, collection=None, model=None) -> list[dict]:
    model = model or _get_model()
    collection = collection or _get_collection()

    query_embedding = model.encode([query]).tolist() if hasattr(model.encode([query]), "tolist") else model.encode([query])
    result = collection.query(query_embeddings=query_embedding, n_results=max(top_k * 2, top_k))

    hits = []
    for id_, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        score = 1 - dist
        if meta.get("page_type") in EVERGREEN_TYPES:
            score += EVERGREEN_BOOST
        hits.append({"id": id_, "text": doc, "metadata": meta, "score": score})

    hits.sort(key=lambda h: h["score"], reverse=True)

    contact_hit = next((h for h in hits if h["metadata"].get("page_type") == "contact"), None)
    top = hits[:top_k]
    if contact_hit and contact_hit["id"] not in {h["id"] for h in top}:
        top = top[: max(top_k - 1, 0)] + [contact_hit]

    return top
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add chatbot/retrieval/__init__.py chatbot/retrieval/retriever.py tests/test_retriever.py
git commit -m "feat: retriever with evergreen boosting and contact-chunk injection"
```

---

### Task 5: LLM provider interface + Ollama implementation

**Files:**
- Create: `chatbot/llm/__init__.py` (empty)
- Create: `chatbot/llm/base.py`
- Create: `chatbot/llm/ollama_provider.py`
- Test: `tests/test_ollama_provider.py`

**Interfaces:**
- Produces: abstract `LLMProvider.generate(system_prompt: str, messages: list[dict], context: str) -> str`; concrete `OllamaProvider(host=OLLAMA_HOST, model=OLLAMA_MODEL)` implementing it, raising `RuntimeError` on connection failure. Consumed by Task 7. A future `GroqProvider` implements the same `generate()` signature — no other file needs to change to add it.

- [ ] **Step 1: Write the failing tests**

`tests/test_ollama_provider.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ollama_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chatbot.llm'`

- [ ] **Step 3: Implement chatbot/llm/base.py**

```python
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
```

- [ ] **Step 4: Implement chatbot/llm/ollama_provider.py**

```python
"""Local Ollama LLM provider - default, zero-cost, offline."""
import requests

from chatbot.llm.base import LLMProvider
from chatbot.config import OLLAMA_HOST, OLLAMA_MODEL


class OllamaProvider(LLMProvider):
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, system_prompt: str, messages: list[dict], context: str) -> str:
        full_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context:\n{context}"},
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ollama_provider.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add chatbot/llm/__init__.py chatbot/llm/base.py chatbot/llm/ollama_provider.py tests/test_ollama_provider.py
git commit -m "feat: swappable LLMProvider interface + local Ollama implementation"
```

---

### Task 6: Prompt builder (persona, guardrails, fee disclaimer)

**Files:**
- Create: `chatbot/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Produces: `SYSTEM_PROMPT: str`; `build_context(hits: list[dict]) -> str`; `has_fee_chunk(hits: list[dict]) -> bool`; `postprocess_answer(answer: str, hits: list[dict]) -> str`. Consumed by Task 7.

- [ ] **Step 1: Write the failing tests**

`tests/test_prompt.py`:
```python
from chatbot.prompt import build_context, has_fee_chunk, postprocess_answer


def test_build_context_includes_type_title_url():
    hits = [{"metadata": {"page_type": "fee", "title": "Whitening", "url": "https://x/fees"}, "text": "£350"}]
    ctx = build_context(hits)
    assert "fee" in ctx and "Whitening" in ctx and "https://x/fees" in ctx and "£350" in ctx


def test_build_context_empty():
    assert build_context([]) == "No relevant information found."


def test_fee_disclaimer_appended_once():
    hits = [{"metadata": {"page_type": "fee"}}]
    answer = postprocess_answer("Whitening costs £350.", hits)
    assert "confirm with the practice" in answer.lower()
    answer2 = postprocess_answer(answer, hits)
    assert answer2.lower().count("confirm with the practice") == 1


def test_no_disclaimer_when_no_fee_chunk():
    hits = [{"metadata": {"page_type": "service"}}]
    answer = postprocess_answer("We offer whitening.", hits)
    assert "confirm with the practice" not in answer.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chatbot.prompt'`

- [ ] **Step 3: Implement chatbot/prompt.py**

```python
"""System prompt (persona + guardrails) and context formatting for the LLM call."""

SYSTEM_PROMPT = """You are the friendly virtual receptionist for Theale Dental Practice.
Answer ONLY using the information given to you in the Context below.
If the answer is not in the Context, say you're not sure and suggest the visitor
contact the practice directly - do not guess or make anything up.
Never give clinical or diagnostic advice; for any medical question, tell the visitor
to book an appointment with the practice.
Keep answers short, warm, and to the point."""

FEE_DISCLAIMER = (
    "\n\n(Prices are as listed on the practice website and may have changed - "
    "please confirm with the practice.)"
)


def build_context(hits: list[dict]) -> str:
    if not hits:
        return "No relevant information found."
    lines = []
    for h in hits:
        meta = h["metadata"]
        lines.append(f"[{meta.get('page_type')}] {meta.get('title')} ({meta.get('url')}):\n{h['text']}")
    return "\n\n".join(lines)


def has_fee_chunk(hits: list[dict]) -> bool:
    return any(h["metadata"].get("page_type") == "fee" for h in hits)


def postprocess_answer(answer: str, hits: list[dict]) -> str:
    if has_fee_chunk(hits) and "confirm with the practice" not in answer.lower():
        answer += FEE_DISCLAIMER
    return answer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompt.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add chatbot/prompt.py tests/test_prompt.py
git commit -m "feat: branded system prompt, grounding guardrails, fee disclaimer"
```

---

### Task 7: FastAPI backend (/chat endpoint, conversation memory)

**Files:**
- Create: `chatbot/api/__init__.py` (empty)
- Create: `chatbot/api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `chatbot.retrieval.retriever.retrieve`, `chatbot.llm.ollama_provider.OllamaProvider`, `chatbot.prompt.{SYSTEM_PROMPT,build_context,postprocess_answer}`
- Produces: FastAPI `app` with `POST /chat {session_id, message} -> {reply, sources}` and `GET /health`. Consumed by Task 8 (widget) and Task 9 (Docker/run).

- [ ] **Step 1: Write the failing tests**

`tests/test_api.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chatbot.api'`

- [ ] **Step 3: Implement chatbot/api/main.py**

```python
"""FastAPI backend exposing POST /chat. Ties retriever + prompt + LLM provider
together and keeps a short per-session conversation history in memory."""
from collections import defaultdict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot.retrieval.retriever import retrieve
from chatbot.llm.ollama_provider import OllamaProvider
from chatbot.prompt import SYSTEM_PROMPT, build_context, postprocess_answer

app = FastAPI(title="Thale Dental Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_provider = OllamaProvider()
_sessions: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY_TURNS = 6


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        return ChatResponse(reply="Please type a question.", sources=[])

    hits = retrieve(message)
    context = build_context(hits)

    history = _sessions[req.session_id]
    history.append({"role": "user", "content": message})
    history = history[-MAX_HISTORY_TURNS * 2:]

    try:
        answer = llm_provider.generate(SYSTEM_PROMPT, history, context)
    except RuntimeError:
        return ChatResponse(
            reply="Sorry, the chatbot is temporarily unavailable. Please try again shortly.",
            sources=[],
        )

    answer = postprocess_answer(answer, hits)
    history.append({"role": "assistant", "content": answer})
    _sessions[req.session_id] = history

    sources = sorted({h["metadata"]["url"] for h in hits})
    return ChatResponse(reply=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add chatbot/api/__init__.py chatbot/api/main.py tests/test_api.py
git commit -m "feat: FastAPI /chat endpoint with session memory and graceful fallback"
```

---

### Task 8: Chat widget (embeddable HTML/CSS/JS)

**Files:**
- Create: `chatbot/widget/widget.html`
- Test: `tests/test_widget_file.py`

**Interfaces:**
- Consumes: `POST {API_URL}/chat {session_id, message} -> {reply, sources}` (Task 7's contract)
- Produces: a single static file droppable into any page (localhost now, WordPress `<script>`/iframe embed later) by changing the `API_URL` constant.

- [ ] **Step 1: Write the failing test**

`tests/test_widget_file.py`:
```python
from pathlib import Path

WIDGET_PATH = Path(__file__).resolve().parent.parent / "chatbot" / "widget" / "widget.html"


def test_widget_file_exists_and_has_api_url_and_chat_endpoint():
    assert WIDGET_PATH.exists()
    content = WIDGET_PATH.read_text(encoding="utf-8")
    assert "API_URL" in content
    assert "/chat" in content
    assert "chat-bubble" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_widget_file.py -v`
Expected: FAIL - file does not exist

- [ ] **Step 3: Implement chatbot/widget/widget.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Theale Dental Chat</title>
<style>
  :root { --primary:#0c6b58; --bg:#ffffff; --text:#1a1a1a; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; margin:0; }
  #chat-bubble {
    position: fixed; bottom: 20px; right: 20px; width: 56px; height: 56px;
    border-radius: 50%; background: var(--primary); color: #fff; border: none;
    font-size: 24px; cursor: pointer; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    z-index: 9999;
  }
  #chat-panel {
    position: fixed; bottom: 88px; right: 20px; width: 340px; max-height: 480px;
    background: var(--bg); border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    display: none; flex-direction: column; overflow: hidden; z-index: 9999;
  }
  #chat-panel.open { display: flex; }
  #chat-header { background: var(--primary); color: #fff; padding: 12px 16px; font-weight: 600; }
  #chat-messages { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 8px; }
  .msg { padding: 8px 12px; border-radius: 10px; max-width: 85%; font-size: 14px; line-height: 1.4; }
  .msg.user { align-self: flex-end; background: var(--primary); color: #fff; }
  .msg.bot { align-self: flex-start; background: #f0f0f0; color: var(--text); }
  #chat-input-row { display: flex; border-top: 1px solid #eee; }
  #chat-input { flex: 1; border: none; padding: 12px; font-size: 14px; }
  #chat-send { border: none; background: var(--primary); color: #fff; padding: 0 16px; cursor: pointer; }
</style>
</head>
<body>

<button id="chat-bubble" aria-label="Open chat">Chat</button>
<div id="chat-panel">
  <div id="chat-header">Theale Dental Assistant</div>
  <div id="chat-messages"></div>
  <div id="chat-input-row">
    <input id="chat-input" type="text" placeholder="Ask a question..." />
    <button id="chat-send">Send</button>
  </div>
</div>

<script>
const API_URL = "http://localhost:8001/chat"; // change to deployed backend URL later

function getSessionId() {
  let id = localStorage.getItem("thale_chat_session");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("thale_chat_session", id);
  }
  return id;
}

const bubble = document.getElementById("chat-bubble");
const panel = document.getElementById("chat-panel");
const messagesEl = document.getElementById("chat-messages");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-send");

bubble.addEventListener("click", () => panel.classList.toggle("open"));

function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, "user");
  input.value = "";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: getSessionId(), message: text }),
    });
    const data = await res.json();
    addMessage(data.reply, "bot");
  } catch (e) {
    addMessage("Sorry, could not reach the chatbot. Is the server running?", "bot");
  }
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });
</script>
</body>
</html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_widget_file.py -v`
Expected: PASS

- [ ] **Step 5: Manual browser verification**

With backend running (Task 9 covers startup), open `chatbot/widget/widget.html` directly in a browser, click the bubble, send "What are your opening hours?", confirm a reply appears and a follow-up question keeps context.

- [ ] **Step 6: Commit**

```bash
git add chatbot/widget/widget.html tests/test_widget_file.py
git commit -m "feat: embeddable chat widget (vanilla HTML/CSS/JS)"
```

---

### Task 9: Docker packaging, ingestion run, end-to-end verification

**Files:**
- Create: `Dockerfile`
- Create: `README_CHATBOT.md`

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: a runnable local stack (`docker run` or direct `uvicorn`) plus a documented manual end-to-end test matching the spec's Testing section.

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["uvicorn", "chatbot.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 2: Create README_CHATBOT.md with setup + run steps**

```markdown
# Thale Dental Chatbot - Local Setup

## Prerequisites
1. Install Ollama: https://ollama.com/download
2. Pull the model: `ollama pull llama3.2:3b`
3. `pip install -r requirements.txt`

## Build the knowledge index (run once, re-run when site/ changes)
`python -m chatbot.ingest.build_index`

## Run the backend
`uvicorn chatbot.api.main:app --reload --port 8001`

## Open the widget
Open `chatbot/widget/widget.html` directly in a browser (Ollama + backend must be running).

## Run tests
`pytest -v`
```

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: All tests from Tasks 1-8 PASS (config, parser, build_index, retriever, ollama_provider, prompt, api, widget_file)

- [ ] **Step 4: Manual end-to-end verification**

1. `ollama pull llama3.2:3b`
2. `python -m chatbot.ingest.build_index` — confirm it prints `Indexed N chunks.` with N > 100
3. `uvicorn chatbot.api.main:app --port 8001` in one terminal
4. `curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -d "{\"session_id\":\"t1\",\"message\":\"how much is teeth whitening?\"}"` — confirm JSON reply mentions whitening price and includes the fee disclaimer
5. `curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -d "{\"session_id\":\"t1\",\"message\":\"who is Dr Bahra?\"}"` — confirm reply describes the doctor
6. Open `chatbot/widget/widget.html` in browser, ask "what are your opening hours?", confirm reply, then ask a follow-up ("and on Saturdays?") and confirm it uses conversation memory

- [ ] **Step 5: Commit**

```bash
git add Dockerfile README_CHATBOT.md
git commit -m "chore: Docker packaging and setup docs for local chatbot stack"
```

---

## Post-plan (documented, not built now)
- Swap `LLM_PROVIDER=groq` + implement `GroqProvider(LLMProvider)` when moving off local-only.
- Deploy backend container to Render/Fly.io free tier; point widget's `API_URL` at it.
- Embed `widget.html`'s markup/script into the real WordPress theme footer.
