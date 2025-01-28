from fastapi.testclient import TestClient
from api.app import app

TEST_API = "/api/censor"
TEST_JSON = {"input_text": "test@example.com"}

client = TestClient(app)

def test_censoring_api():
    response = client.post(TEST_API, json=TEST_JSON)
    assert response.json()["test_sentence"] == "[CENSORED]"