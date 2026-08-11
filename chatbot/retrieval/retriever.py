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

    embedded = model.encode([query])
    query_embedding = embedded.tolist() if hasattr(embedded, "tolist") else list(embedded)
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
    top = hits[:top_k]

    if any(h["metadata"].get("page_type") == "contact" for h in top):
        return top

    # Contact wasn't among the ANN top-K window (can happen on any query, not
    # just unrelated ones, once the collection is large) - the contact chunk
    # must always be available regardless, so fetch it directly by metadata
    # instead of relying on it having surfaced in the similarity search.
    contact_hit = next((h for h in hits if h["metadata"].get("page_type") == "contact"), None)
    if contact_hit is None:
        fetched = collection.get(where={"page_type": "contact"}, limit=1)
        if fetched["ids"]:
            contact_hit = {
                "id": fetched["ids"][0],
                "text": fetched["documents"][0],
                "metadata": fetched["metadatas"][0],
                "score": 0.0,
            }

    if contact_hit is not None:
        top = top[: max(top_k - 1, 0)] + [contact_hit]

    return top
