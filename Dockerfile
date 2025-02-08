# 기본 Python 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements_Docker.txt .

# 패키지 설치
RUN pip install --no-cache-dir -r requirements_Docker.txt
RUN pip list

# 애플리케이션 코드 복사
COPY . .

# FastAPI 서버 실행 (Uvicorn)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
