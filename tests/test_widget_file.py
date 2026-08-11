from pathlib import Path

WIDGET_PATH = Path(__file__).resolve().parent.parent / "chatbot" / "widget" / "widget.html"


def test_widget_file_exists_and_has_api_url_and_chat_endpoint():
    assert WIDGET_PATH.exists()
    content = WIDGET_PATH.read_text(encoding="utf-8")
    assert "API_URL" in content
    assert "/chat" in content
    assert "chat-bubble" in content
