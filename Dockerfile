# CardioGuard-AI Docker Image (multi-stage: frontend + API)
# docker-compose up --build → http://localhost:8000

# --- Stage 1: Frontend build ---
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: API ---
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY checkpoints/ checkpoints/
COPY logs/ logs/
COPY artifacts/ artifacts/
COPY sample.npy test_mi_sample.npz ./

COPY --from=frontend-build /app/frontend/dist frontend/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
