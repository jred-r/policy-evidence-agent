FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
COPY data ./data
COPY eval ./eval

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"
EXPOSE 8000

CMD ["uvicorn", "qa.a2a_app:app", "--host", "0.0.0.0", "--port", "8000"]
