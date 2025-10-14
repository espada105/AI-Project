# Python 3.11 slim 이미지 사용
FROM python:3.11-slim

# 메타데이터 추가
LABEL maintainer="your-email@example.com"
LABEL description="간단한 Python 웹 애플리케이션"

# 작업 디렉토리 설정
WORKDIR /app

# Python 출력 버퍼링 비활성화 (로그 실시간 확인)
ENV PYTHONUNBUFFERED=1

# 애플리케이션 파일 복사
COPY app.py .

# 애플리케이션이 사용할 포트 노출
EXPOSE 8000

# 환경변수 설정
ENV APP_NAME="도커 Python 앱"
ENV PORT=8000

# 컨테이너 실행 시 기본 명령어
CMD ["python", "app.py"]
