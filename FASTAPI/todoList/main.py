from fastapi import FastAPI, HTTPException # 'FastAPI' 클래스는 대문자로 임포트
from pydantic import BaseModel
from typing import Optional, List, Dict

# --- 데이터 모델 ---
class TodoItem(BaseModel):
    id: int
    title: str
    is_completed: bool = False

class TodoCreate(BaseModel):
    title: str
    is_completed: bool = False

# --- 애플리케이션 초기화 ---
app = FastAPI() # 'FastAPI' 클래스 사용

# --- 임시 데이터 저장소 초기화 ---
next_id = 1 # 1. next_id를 먼저 초기화합니다.
todos: Dict[int, TodoItem] = {}

# next_id += 1 <- 이 줄은 삭제되어야 합니다. 초기값을 1로 설정했으므로

# ------------------------------------
# --- 경로 동작 함수 (API Endpoints) ---
# ------------------------------------

# 1. To-Do 항목 생성 (POST)
@app.post("/todos/", response_model=TodoItem)
def create_todo(todo: TodoCreate):
    global next_id
    # 생성할 때 next_id를 사용하고,
    new_todo = TodoItem(id=next_id, title=todo.title, is_completed=todo.is_completed)
    todos[next_id] = new_todo
    # 다음 사용을 위해 next_id를 증가시킵니다.
    next_id += 1 
    return new_todo

# 2. 모든 To-Do 항목 조회 (GET)
@app.get("/todos/", response_model=List[TodoItem])
def read_todos():
    return list(todos.values())

# 3. 특정 To-Do 항목 조회 (GET)
@app.get("/todos/{todo_id}", response_model=TodoItem)
def read_todo(todo_id: int):
    if todo_id not in todos:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todos[todo_id]

# 4. 특정 To-Do 항목 수정 (PUT)
@app.put("/todos/{todo_id}", response_model=TodoItem)
def update_todo(todo_id: int, todo: TodoCreate):
    if todo_id not in todos:
        raise HTTPException(status_code=404, detail="Todo not found")
    
    # 기존 항목을 업데이트 (ID는 그대로 유지)
    updated_todo = TodoItem(id=todo_id, title=todo.title, is_completed=todo.is_completed)
    todos[todo_id] = updated_todo
    return updated_todo

# 5. 특정 To-Do 항목 삭제 (DELETE)
@app.delete("/todos/{todo_id}", status_code=204) 
def delete_todo(todo_id: int):
    if todo_id not in todos:
        raise HTTPException(status_code=404, detail="Todo not found")
    
    del todos[todo_id]
    return # 204 응답에는 본문이 없으므로 명시적인 return 값 없이도 잘 작동합니다.