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
    # MiniLM embeddings are meant for cosine similarity - Chroma's default
    # (L2) on these gives distances >1, making the retriever's 1-dist score
    # uninterpretable/poorly calibrated. Cosine keeps scores in a sane range.
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

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
