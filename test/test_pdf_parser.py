import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Patch fitz and openai before importing the app
_mock_doc = MagicMock()
_mock_page = MagicMock()
_mock_pixmap = MagicMock()
_mock_pixmap.tobytes.return_value = b"fake-png-bytes"
_mock_page.get_pixmap.return_value = _mock_pixmap
_mock_doc.__len__ = MagicMock(return_value=2)
_mock_doc.__iter__ = MagicMock(return_value=iter([_mock_page, _mock_page]))
_mock_doc.__getitem__ = MagicMock(side_effect=lambda i: _mock_page)
_mock_doc.close = MagicMock()

_mock_glm_response = MagicMock()
_mock_glm_response.choices = [MagicMock(message=MagicMock(content="page text"))]

_mock_glm_client = MagicMock()
_mock_glm_client.chat = MagicMock()
_mock_glm_client.chat.completions = MagicMock()
_mock_glm_client.chat.completions.create = AsyncMock(return_value=_mock_glm_response)
_mock_glm_client.close = AsyncMock()

with (
    patch("fitz.open", return_value=_mock_doc),
    patch("openai.AsyncOpenAI", return_value=_mock_glm_client),
):
    from fastapi.testclient import TestClient
    from pdf_parser.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_parse_returns_text():
    pdf_b64 = base64.b64encode(b"%PDF-fake").decode()
    resp = client.post("/parse", json={"pdf_bytes": pdf_b64, "filename": "test.pdf"})
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    # Two pages joined with \n\n
    assert data["text"] == "page text\n\npage text"


def test_parse_invalid_pdf():
    with patch("fitz.open", side_effect=Exception("bad pdf")):
        pdf_b64 = base64.b64encode(b"not a pdf").decode()
        resp = client.post("/parse", json={"pdf_bytes": pdf_b64, "filename": "bad.pdf"})
    assert resp.status_code == 400
