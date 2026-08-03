# Multi-stage so the runtime image carries no build tooling.
FROM python:3.12-slim AS build

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
# --no-cache-dir keeps the wheel cache out of the layer; the venv is what gets copied on.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim

# Never run as root. A container that can write to its own filesystem is one
# prompt injection away from being a different program.
RUN useradd --create-home --uid 10001 app
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
WORKDIR /home/app

# Proves the package imports without needing a key or a network call.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD python -c "import agentapp; print(agentapp.__version__)" || exit 1

ENTRYPOINT ["agentapp"]
CMD ["--offline", "what is the weather in Bergen?"]
