# CardioGuard API only (Render). Frontend → Vercel.
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
COPY features_out/ features_out/

ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV ALLOW_CLIENT_LLM_KEY=0
ENV ENABLE_DEBUG_ENDPOINTS=0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT:-8000}/health')"

CMD ["sh", "-c", "uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
