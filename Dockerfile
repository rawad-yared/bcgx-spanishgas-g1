FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY src ./src
COPY configs ./configs
COPY pyproject.toml README.md ./

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      fastapi \
      "uvicorn[standard]" \
      pydantic \
      pyyaml \
      numpy \
      scikit-learn \
      boto3

EXPOSE 8000

CMD ["uvicorn", "src.serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
