from fastapi.testclient import TestClient
from eye_diseases_classification.api import app

def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the Eye Disease Classification Model API!"}
        
        
def test_read_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "checkpoint" in data
        assert data["checkpoint"] is not None