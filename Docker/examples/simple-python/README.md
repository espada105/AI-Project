# 간단한 Python 도커 예제

## 📝 설명
순수 Python으로 만든 간단한 웹서버를 도커로 실행하는 예제입니다.

## 🚀 실행 방법

### 1. 이미지 빌드
```bash
cd Docker/examples/simple-python
docker build -t simple-python-app .
```

### 2. 컨테이너 실행
```bash
# 기본 실행
docker run -p 8000:8000 simple-python-app

# 백그라운드 실행
docker run -d -p 8000:8000 --name my-python-app simple-python-app

# 환경변수와 함께 실행
docker run -p 8000:8000 -e APP_NAME="내 파이썬 앱" simple-python-app
```

### 3. 접속 확인
- 메인 페이지: http://localhost:8000
- API 정보: http://localhost:8000/api/info
- 헬스체크: http://localhost:8000/api/health

### 4. 컨테이너 정리
```bash
# 컨테이너 중지 및 삭제
docker stop my-python-app
docker rm my-python-app

# 이미지 삭제
docker rmi simple-python-app
```

## 🔍 학습 포인트

1. **FROM**: 기본 이미지 선택
2. **WORKDIR**: 작업 디렉토리 설정
3. **COPY**: 파일 복사
4. **ENV**: 환경변수 설정
5. **EXPOSE**: 포트 문서화
6. **CMD**: 실행 명령어

## 💡 실습 과제

1. 포트를 3000으로 변경해보세요
2. 새로운 API 엔드포인트를 추가해보세요
3. 환경변수를 더 추가해보세요
