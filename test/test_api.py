from unittest.mock import MagicMock, AsyncMock, patch

# Build mock instances that lifespan will receive from patched constructors
_mock_query = MagicMock()
_mock_query.setup = AsyncMock()
_mock_query.query_search = AsyncMock(return_value="mock context")
_mock_query.create_embeddings = AsyncMock(return_value="test-doc-id")
_mock_query.async_client = AsyncMock()

_mock_bot = MagicMock()
_mock_bot.generate_response = AsyncMock(return_value="mock answer")
_mock_bot.stream_response = AsyncMock(return_value=aiter([]))


async def aiter(items):
    for item in items:
        yield item


# Patch constructors so lifespan receives our mocks instead of real objects.
# These patches stay active for the entire test module.
with (
    patch("api.Query", return_value=_mock_query),
    patch("api.Chatbot", return_value=_mock_bot),
):
    from fastapi.testclient import TestClient
    from api import app

client = TestClient(app)


def test_chat_response():
    _mock_query.query_search.return_value = "some context"
    _mock_bot.generate_response.return_value = "some answer"

    response = client.post("/chat_response", json={"text": "what is cheese making?"})

    assert response.status_code == 200
    assert "response" in response.json()


def test_chat_response_with_doc_id():
    _mock_query.query_search.return_value = "filtered context"
    _mock_bot.generate_response.return_value = "filtered answer"

    response = client.post(
        "/chat_response",
        json={"text": "what is this about?", "doc_id": "abc-123"},
    )

    assert response.status_code == 200
    _mock_query.query_search.assert_called_with("what is this about?", doc_id="abc-123")


def test_create_embeddings_returns_doc_id():
    _mock_query.create_embeddings.return_value = "generated-doc-id"

    response = client.post(
        "/create_embeddings",
        json={"text": "some document text", "filename": "test.txt"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["create_embeddings"] is True
    assert data["doc_id"] == "generated-doc-id"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_pdf():
    _mock_query.create_embeddings.return_value = "pdf-doc-id"

    mock_parse_resp = MagicMock()
    mock_parse_resp.json.return_value = {"text": "extracted pdf text"}
    mock_parse_resp.raise_for_status = MagicMock()

    mock_http_client = MagicMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_parse_resp)

    with patch("api.httpx.AsyncClient", return_value=mock_http_client):
        response = client.post(
            "/upload_pdf",
            files={"file": ("test.pdf", b"%PDF-fake", "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["create_embeddings"] is True
    assert data["doc_id"] == "pdf-doc-id"
    _mock_query.create_embeddings.assert_called_with("extracted pdf text", "test.pdf")
