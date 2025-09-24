from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import time

app = FastAPI()

# ----------------
# 가상의 데이터 저장소
items_db = {}
# ----------------

# 2.2 의존성 주입 (이전과 동일)
def get_db_session():
    try:
        print("DB 세션 시작")
        yield "가상 DB 세션"
    finally:
        print("DB 세션 종료")

# 2.3 미들웨어 (이전과 동일)
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"요청 처리 시간: {process_time:.4f}초")
    return response

# Pydantic 모델 (이전과 동일)
class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None

# ----------------
# CRUD 엔드포인트 구현
# ----------------

# C: Create (생성)
@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item, db: str = Depends(get_db_session)):
    item_id = len(items_db) + 1
    items_db[item_id] = item
    print(f"DB 세션 사용: {db}")
    return {"message": "아이템이 성공적으로 생성되었습니다.", "item_id": item_id, "item": item}

# R: Read (조회 - 단일)
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"아이템 ID {item_id}을 찾을 수 없습니다."
        )
    return {"item_id": item_id, "item": items_db[item_id]}

# U: Update (수정)
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"아이템 ID {item_id}을 찾을 수 없습니다."
        )
    items_db[item_id] = item
    return {"message": f"아이템 ID {item_id}가 업데이트되었습니다."}

# D: Delete (삭제)
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"아이템 ID {item_id}을 찾을 수 없습니다."
        )
    del items_db[item_id]
    return {"message": f"아이템 ID {item_id}가 삭제되었습니다."}