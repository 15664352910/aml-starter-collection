FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    MEMORY_DB_PATH=/app/chroma_data \
    EMBEDDING_BACKEND=st \
    EMBEDDING_MODEL=all-MiniLM-L6-v2

# 国内构建时可换镜像：docker build --build-arg HF_ENDPOINT=https://hf-mirror.com .
ARG HF_ENDPOINT=https://huggingface.co
ENV HF_ENDPOINT=${HF_ENDPOINT}

WORKDIR /app

COPY requirements.txt .
# CPU 版 torch，镜像更小；构建期预下载嵌入模型，运行时离线可用
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY . .
RUN mkdir -p /app/chroma_data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
