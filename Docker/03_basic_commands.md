# 도커 기본 명령어 가이드

## 🚀 이미지 관련 명령어

### 이미지 다운로드
```bash
# Docker Hub에서 이미지 받기
docker pull python:3.11
docker pull nginx:latest
docker pull mysql:8.0
```

### 이미지 빌드
```bash
# Dockerfile로 이미지 만들기
docker build -t my-app:v1.0 .
docker build -t my-app:latest --file Dockerfile.prod .

# 태그 없이 빌드
docker build .
```

### 이미지 목록 확인
```bash
# 모든 이미지 보기
docker images

# 특정 이미지만 검색
docker images python
```

### 이미지 삭제
```bash
# 특정 이미지 삭제
docker rmi python:3.11
docker rmi image_id

# 사용하지 않는 이미지 모두 삭제
docker image prune
```

## 📦 컨테이너 관련 명령어

### 컨테이너 실행
```bash
# 기본 실행
docker run python:3.11

# 백그라운드 실행 (-d)
docker run -d nginx

# 포트 연결 (-p)
docker run -p 8080:80 nginx

# 이름 지정 (--name)
docker run --name my-nginx -p 8080:80 nginx

# 환경변수 설정 (-e)
docker run -e MYSQL_ROOT_PASSWORD=secret mysql:8.0

# 볼륨 마운트 (-v)
docker run -v /host/path:/container/path nginx

# 인터랙티브 모드 (-it)
docker run -it python:3.11 bash
```

### 실행 중인 컨테이너 확인
```bash
# 실행 중인 컨테이너만
docker ps

# 모든 컨테이너 (중지된 것 포함)
docker ps -a

# 컨테이너 ID만 표시
docker ps -q
```

### 컨테이너 제어
```bash
# 컨테이너 중지
docker stop container_name
docker stop container_id

# 컨테이너 시작
docker start container_name

# 컨테이너 재시작
docker restart container_name

# 컨테이너 일시정지/재개
docker pause container_name
docker unpause container_name
```

### 컨테이너 삭제
```bash
# 특정 컨테이너 삭제
docker rm container_name

# 실행 중인 컨테이너 강제 삭제
docker rm -f container_name

# 중지된 컨테이너 모두 삭제
docker container prune
```

## 🔍 정보 확인 명령어

### 컨테이너 상세 정보
```bash
# 컨테이너 상세 정보
docker inspect container_name

# 컨테이너 로그 확인
docker logs container_name
docker logs -f container_name  # 실시간 로그

# 컨테이너 내부 접속
docker exec -it container_name bash
docker exec -it container_name sh
```

### 리소스 사용량
```bash
# 실시간 리소스 사용량
docker stats

# 특정 컨테이너만
docker stats container_name
```

## 🧹 정리 명령어

### 시스템 정리
```bash
# 사용하지 않는 모든 것 삭제
docker system prune

# 더 강력한 정리 (볼륨까지)
docker system prune -a --volumes

# 디스크 사용량 확인
docker system df
```

## 📝 실습 예제

### 1. 간단한 웹서버 실행
```bash
# nginx 웹서버 실행
docker run -d --name my-web -p 8080:80 nginx

# 브라우저에서 http://localhost:8080 접속
# 컨테이너 중지 및 삭제
docker stop my-web
docker rm my-web
```

### 2. Python 개발환경
```bash
# Python 컨테이너에서 인터랙티브 세션
docker run -it --name python-dev python:3.11 bash

# 컨테이너 내부에서
pip install requests
python -c "import requests; print(requests.get('https://httpbin.org/json').json())"

# 컨테이너 나가기
exit
```

### 3. 데이터베이스 실행
```bash
# MySQL 데이터베이스 실행
docker run -d \
  --name mysql-db \
  -e MYSQL_ROOT_PASSWORD=mypassword \
  -e MYSQL_DATABASE=testdb \
  -p 3306:3306 \
  mysql:8.0

# 데이터베이스 접속
docker exec -it mysql-db mysql -u root -p
```

## 💡 유용한 팁

### 1. 한 줄로 컨테이너 정리
```bash
# 모든 컨테이너 중지 후 삭제
docker stop $(docker ps -q) && docker rm $(docker ps -aq)
```

### 2. 이미지 태그 관리
```bash
# 같은 이미지에 여러 태그
docker tag my-app:v1.0 my-app:latest
docker tag my-app:v1.0 my-registry.com/my-app:v1.0
```

### 3. 환경별 설정 파일 사용
```bash
# docker-compose.yml 사용 (권장)
docker-compose up -d
docker-compose down
```
