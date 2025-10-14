# 1. Python이 설치된 기본 상자를 가져옵니다
FROM python:3.11-slim

# 2. 상자 안에서 작업할 폴더를 만듭니다
WORKDIR /app

# 3. 우리가 만든 hello.py 파일을 상자 안으로 복사합니다
COPY hello.py .

# 4. 상자가 시작될 때 hello.py를 실행하도록 설정합니다
CMD ["python", "hello.py"]
