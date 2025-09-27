# test_main.py
from fastapi.testclient import TestClient
from main import app, items_db  # main.py에서 app과 items_db를 임포트

# TestClient 객체 생성
client = TestClient(app)

# 통합 테스트: 아이템 생성 -> 조회 -> 수정 -> 삭제
def test_crud_flow():
    # 1. 아이템 생성 (POST)
    response = client.post(
        "/items/",
        json={"name": "테스트 아이템", "price": 10.99}
    )
    assert response.status_code == 201
    item_data = response.json()
    assert item_data["item"]["name"] == "테스트 아이템"
    item_id = item_data["item_id"]

    # 2. 아이템 조회 (GET)
    response = client.get(f"/items/{item_id}")
    assert response.status_code == 200
    assert response.json()["item"]["name"] == "테스트 아이템"

    # 3. 아이템 수정 (PUT)
    response = client.put(
        f"/items/{item_id}",
        json={"name": "수정된 아이템", "price": 15.00}
    )
    assert response.status_code == 200
    assert response.json()["message"] == f"아이템 ID {item_id}가 업데이트되었습니다."

    # 4. 아이템 삭제 (DELETE)
    response = client.delete(f"/items/{item_id}")
    assert response.status_code == 200
    assert response.json()["message"] == f"아이템 ID {item_id}가 삭제되었습니다."

    # 5. 삭제 후 조회 시 404 확인
    response = client.get(f"/items/{item_id}")
    assert response.status_code == 404
    
# 단위 테스트: 특정 오류 상황 테스트
def test_read_non_existent_item():
    response = client.get("/items/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "아이템 ID 999을 찾을 수 없습니다."