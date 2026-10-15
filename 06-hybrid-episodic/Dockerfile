# Pinned to 3.12 rather than latest: the pinned dependency set has prebuilt wheels
# for 3.12, so this build never needs a Rust or C++ toolchain. (On 3.14,
# pydantic-core has no wheel and falls back to a source build.)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    EMBED_ENABLED=true \
    MEM0_ENABLED=false

WORKDIR /app

# CPU-only torch. The default index serves the CUDA build, which adds several GB
# for no benefit here.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# Bake the encoder into the image so startup needs no network and the deployed
# system is byte-identical to what was tested.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBED_MODEL}', device='cpu')"

COPY src ./src

EXPOSE 8000

# Unauthenticated GET /health, any 2xx — what the platform probes when no separate
# health URL is configured.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if 200 <= urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status < 300 else 1)"

# Single process on purpose. The index lives in process memory, so extra workers
# would split one user_id's memories across processes and each Search would see
# only its own shard — a silent recall loss that raises no error. Do not set
# WEB_CONCURRENCY without first moving the index out of process.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
