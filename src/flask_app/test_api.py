from fastapi.testclient import TestClient
from mistral_model import Chatbot
import pytest
from flask_chat import app

client = TestClient(app)
bot = Chatbot()

def test_chat_response():
    response = client.post(
        "/chat_response",
        json={"text": "Hello, World!", "context": "Greetings"},
    )
    assert response.status_code == 200
    assert "response" in response.json()

@pytest.fixture
def mock_generate_response(monkeypatch):
    def mock_response(*args, **kwargs):
        return "Mocked chatbot response"
    monkeypatch.setattr(bot, "generate_response", mock_response)

def test_chat_response_with_mock(mock_generate_response):
    response = client.post(
        "/chat_response",
        json={"text": "Hello, World!", "context": "Greetings"},
    )
    assert response.status_code == 200
    assert response.json() == {"response": "Mocked chatbot response"}
