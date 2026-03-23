from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.models import SparseVector

from src.text_generation.embedding_client import RemoteDenseEmbeddings, RemoteSparseEmbeddings


def _make_mock_response(json_data: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=json_data)
    return mock_resp


@pytest.mark.asyncio
async def test_dense_aembed_documents_returns_list_of_floats():
    expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_response = _make_mock_response({"embeddings": expected})

    client = RemoteDenseEmbeddings(base_url="http://localhost:7000")
    with patch.object(client._async_client, "post", new=AsyncMock(return_value=mock_response)):
        result = await client.aembed_documents(["hello", "world"])

    assert result == expected
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], float)
    await client.close()


@pytest.mark.asyncio
async def test_dense_aembed_query_returns_single_vector():
    expected_vector = [0.1, 0.2, 0.3]
    mock_response = _make_mock_response({"embeddings": [expected_vector]})

    client = RemoteDenseEmbeddings(base_url="http://localhost:7000")
    with patch.object(client._async_client, "post", new=AsyncMock(return_value=mock_response)):
        result = await client.aembed_query("hello")

    assert result == expected_vector
    assert isinstance(result, list)
    assert isinstance(result[0], float)
    await client.close()


@pytest.mark.asyncio
async def test_sparse_aembed_documents_returns_sparse_vectors():
    raw = [
        {"indices": [1, 5, 10], "values": [0.9, 0.5, 0.1]},
        {"indices": [2, 7], "values": [0.8, 0.3]},
    ]
    mock_response = _make_mock_response({"embeddings": raw})

    client = RemoteSparseEmbeddings(base_url="http://localhost:7000")
    with patch.object(client._async_client, "post", new=AsyncMock(return_value=mock_response)):
        result = await client.aembed_documents(["hello", "world"])

    assert len(result) == 2
    assert isinstance(result[0], SparseVector)
    assert result[0].indices == [1, 5, 10]
    assert result[0].values == [0.9, 0.5, 0.1]
    assert isinstance(result[1], SparseVector)
    await client.close()


@pytest.mark.asyncio
async def test_sparse_aembed_query_returns_single_sparse_vector():
    raw = [{"indices": [3, 8], "values": [0.7, 0.4]}]
    mock_response = _make_mock_response({"embeddings": raw})

    client = RemoteSparseEmbeddings(base_url="http://localhost:7000")
    with patch.object(client._async_client, "post", new=AsyncMock(return_value=mock_response)):
        result = await client.aembed_query("test query")

    assert isinstance(result, SparseVector)
    assert result.indices == [3, 8]
    assert result.values == [0.7, 0.4]
    await client.close()
