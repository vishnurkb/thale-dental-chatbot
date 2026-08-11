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
