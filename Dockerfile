# The replay service and HUD. Builds the synthetic demo on first start if no
# models exist, so `docker compose up` works with no API access.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.11.32

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    F1PRED_DATA_DIR=/app/data

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group model --group serve --group transform --no-install-project

COPY src ./src
COPY transform ./transform
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN uv sync --frozen --no-dev --group model --group serve --group transform \
    && useradd --create-home --uid 1000 f1 \
    && mkdir -p /app/data \
    && chown -R f1:f1 /app \
    && chmod +x /usr/local/bin/entrypoint.sh

USER f1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/api/health')"

ENTRYPOINT ["entrypoint.sh"]
