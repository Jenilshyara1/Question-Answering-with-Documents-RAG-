from fastapi.testclient import TestClient
from api import app
client = TestClient(app)

def test_chat_response():
    response = client.post(
        "/chat_response",
        json={"text": "what is cheese making?", "context": "Cheesemaking is the process of turning milk into a semisolid mass. This is done by using a coagulating agent, such as rennet, acid, heat, or a combination of these"},
    )
    assert response.status_code == 200
    assert "response" in response.json()
