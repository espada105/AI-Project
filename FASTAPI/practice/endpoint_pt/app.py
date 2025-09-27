# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 사용할 모델 이름 설정
model_name = "beomi/KoAlpaca-Polyglot-5.8B"

# FastAPI 앱 생성
app = FastAPI()

# 모델과 토크나이저를 전역 변수로 선언
tokenizer = None
model = None

# Pydantic을 사용해 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    prompt: str

# 앱 시작 시 모델과 토크나이저를 메모리에 로드
@app.on_event("startup")
def load_model():
    global tokenizer, model
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"모델을 {device}에 로딩 중...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 모델을 선택된 장치로 이동
    model.to(device)
    model.eval()
    
    print("모델 로딩 완료!")

# 챗봇 응답을 위한 POST 엔드포인트
@app.post("/chat")
def get_chat_response(request: ChatRequest):
    if not model or not tokenizer:
        return {"error": "모델이 아직 로드되지 않았습니다."}

    # 사용자 프롬프트
    user_prompt = request.prompt
    
    # 모델 입력 형식에 맞게 프롬프트 구성
    input_text = f"### 질문: {user_prompt}\n\n### 답변: "
    
    # 토큰화 및 모델 입력 생성
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    # 모델이 텍스트 생성
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    # 생성된 텍스트 디코딩
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 모델 답변 부분만 추출 (이전 질문/답변 형식 제거)
    response_parts = response_text.split("### 답변: ")
    if len(response_parts) > 1:
        clean_response = response_parts[1].strip()
    else:
        clean_response = response_text.strip()
    
    return {"response": clean_response}

# 앱 시작 시 다음 명령어를 실행하세요: uvicorn app:app --reload