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
