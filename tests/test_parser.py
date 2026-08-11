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
