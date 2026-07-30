FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/crop_yield_random_forest.joblib \
    MODEL_METADATA_PATH=/app/artifacts/model_metadata.json

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir \
    fastapi>=0.115,<1.0 \
    "uvicorn[standard]>=0.30,<1.0" \
    pandas>=2.2,<3.0 \
    numpy>=2.0,<3.0 \
    scikit-learn>=1.5,<2.0 \
    joblib>=1.4,<2.0 \
    && pip install --no-cache-dir .

COPY artifacts ./artifacts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "crop_yield.api:app", "--host", "0.0.0.0", "--port", "8000"]
