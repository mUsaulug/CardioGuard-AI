# ====== Stage 1: Frontend Build ======
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ====== Stage 2: Python Backend ======
FROM python:3.10-slim
WORKDIR /app

# System deps (libgomp for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

# Python deps (CPU-only PyTorch for smaller image)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ src/
COPY checkpoints/ checkpoints/
COPY logs/ logs/
COPY artifacts/ artifacts/
COPY sample.npy test_mi_sample.npz ./

# Frontend static files from build stage
COPY --from=frontend-builder /app/frontend/dist frontend/dist

EXPOSE 8000

CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
