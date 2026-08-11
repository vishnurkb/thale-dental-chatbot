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
