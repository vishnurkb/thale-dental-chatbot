# Thale Dental Site Chatbot — RAG Architecture Design

## Goal
Free, fully local, fully working chatbot answering visitor questions using content
scraped from `site/` (109-page static mirror of thealedental.co.uk). Prototype now
on localhost; architected so it can later move to a real deployment (site embed +
optional cloud LLM) via config change, not rewrite.

## Constraints
- Zero cost. No paid API, no paid hosting for the prototype.
- Runs on user's laptop: local LLM via Ollama, local embeddings, local vector DB.
- Q&A only — no bookings, no external actions, no PMS/calendar integration.
- Deployment-friendly: components decoupled so LLM provider and hosting can change
  independently later (documented as Phase 2, not built now).

## Architecture

Two lanes (see architecture diagram presented in chat):

**Ingestion (offline, run once / on content update)**
`site/*.html` → Parser+Chunker → Local Embedder → Vector Store (Chroma, on disk)

**Query (runtime, every chat message)**
Chat Widget → API Backend (FastAPI) → Retriever (reads Vector Store) → LLM Provider
(Ollama, local) → answer streams back to widget

## Components

| Component | Responsibility | Tech |
|---|---|---|
| Parser+Chunker | Strip HTML → clean text → structure-aware chunks | Python, BeautifulSoup |
| Local Embedder | Turn chunk text into vectors | `sentence-transformers` (all-MiniLM-L6-v2), CPU |
| Vector Store | Persist chunks + vectors, similarity search | ChromaDB (local, file-based) |
| Retriever | Embed query, top-k similarity search, apply boosts | Python, in backend |
| LLM Provider | Generate grounded answer from context + query | `LLMProvider` interface; `OllamaProvider` (default) |
| API Backend | HTTP endpoint tying retriever + LLM together | FastAPI + uvicorn |
| Chat Widget | User-facing chat UI | Vanilla HTML/CSS/JS, no framework |
| Config | All tunables in one place, env-driven | `.env` / `config.py` |

## Domain-specific customizations (not generic RAG)

1. **Structure-aware chunking**: FAQ page chunked per Q&A pair (not fixed-size
   splits). `our-fees` parsed per treatment row so price stays attached to correct
   treatment. Doctor bio pages = one chunk per doctor.
2. **Per-chunk metadata**: `{page_type: faq|service|fee|doctor|contact|blog, title,
   url, date}`. Enables source citation and relevance boosting.
3. **Relevance boosting**: evergreen page types (`service`, `faq`, `fee`, `contact`,
   `doctor`) get a small score boost over stale 2013-2020 blog/news posts on
   borderline matches — user chose to keep all 109 pages including old news, so this
   keeps answers focused without discarding data.
4. **Always-available contact chunk**: address/phone/opening-hours injected into every
   prompt's context directly, not left to similarity search (near-universal query).
5. **Grounding guardrails**: system prompt restricts answers to retrieved context
   only; explicit "not sure — contact practice" fallback when no good match; never
   gives clinical/diagnostic advice, redirects medical questions to booking a real
   appointment; auto-appends a disclaimer on fee-related answers (prices may be
   stale since capture).
6. **Branded persona**: system prompt gives the bot a consistent Thale Dental
   receptionist-style voice, not generic assistant tone.
7. **Multi-turn memory**: short rolling conversation history included in prompt so
   follow-up questions work.

## Deployment-friendly design

- `LLMProvider` is an interface. `OllamaProvider` implements it today; a future
  `GroqProvider` (or other free-tier cloud API) can implement the same interface.
  Switching is one env var (`LLM_PROVIDER=ollama|groq`) — zero backend logic changes.
- Ingestion and Serving are separate processes. Re-run ingestion only when site
  content changes; backend just reads the persisted Chroma DB and stays lightweight.
- Backend packaged as a Docker container (FastAPI + uvicorn) from the start —
  `docker run` locally now, same image deploys unchanged to a free host (Render/
  Fly.io) later if going live. Note: Vercel serverless cannot run Ollama (stateful
  local model needs a real host); if a cloud LLM provider is swapped in later, the
  backend becomes stateless and Vercel becomes viable too.
- Chat widget is a single vanilla JS/CSS file — same file works on localhost now and
  drops into WordPress later via a `<script>` tag, just pointing `API_URL` elsewhere.
- All tunables (chunk size, top-k, model names, ports, boost weights) live in one
  `.env` / `config.py`.

## Error handling

- Ollama not running / model not pulled → backend returns clear error to widget
  ("chatbot temporarily unavailable"), logs actionable message to console.
- No relevant chunks found (low similarity scores) → bot replies with fallback,
  does not hallucinate an answer.
- Malformed/empty user input → widget-side validation, backend rejects empty POSTs.
- Ingestion errors on a single page (malformed HTML) → log and skip that page,
  don't abort the whole ingestion run.

## Testing

- Ingestion: run on `site/`, assert chunk count > 0, assert FAQ/fees/doctor pages
  produce structured chunks (not just generic splits).
- Retrieval: fixed test queries ("teeth whitening cost", "who is Dr Bahra", "opening
  hours") → assert top result is the expected page type.
- End-to-end: start backend + Ollama, POST sample questions to `/chat`, assert
  non-empty grounded answer, assert fee questions include disclaimer.
- Manual: open widget in browser, verify chat round-trip, verify follow-up question
  uses memory.

## Out of scope (for this build)
- Real appointment booking / PMS integration
- Cloud LLM wiring (interface ready, provider not implemented yet)
- Production deployment / WordPress embed (widget built embed-ready, not embedded)
