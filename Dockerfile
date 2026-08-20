# NewbieDuo -- CKD guideline evidence service.
#
# Two stages on purpose. The builder installs the toolchain needed to compile
# wheels; the runtime carries only the interpreter, the installed packages and
# the source. It keeps the image an operator has to accept onto a clinical
# network as small as the dependency set allows.
#
# CPU-only torch is deliberate. One warm container serves a whole site -- the
# five CDS trigger questions are fixed, so their answers are computed once at
# startup and served from cache thereafter -- and a CUDA image would triple the
# size to accelerate work that is already done before the first patient.

# ───────────────────────────────────────────────────────────── builder
FROM python:3.10-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# The CPU wheel index has to come first, or pip resolves the default CUDA build
# of torch and the image grows by roughly two gigabytes for no benefit here.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# ───────────────────────────────────────────────────────────── runtime
FROM python:3.10-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="NewbieDuo" \
      org.opencontainers.image.description="Guideline-grounded CKD decision support with CDS Hooks" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Keep every model download inside one mountable directory, so an air-gapped
    # site can populate it once and run offline thereafter.
    HF_HOME=/data/.cache/huggingface \
    NEWBIEDUO_HOST=0.0.0.0 \
    NEWBIEDUO_PORT=8000

# libgomp is required by torch; curl is used by the healthcheck below.
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY newbieduo/ ./newbieduo/
COPY scripts/ ./scripts/
COPY eval/ ./eval/
COPY README.md LICENSE NOTICE ./

# Run unprivileged. /data is the only writable path the service needs, and it is
# where the operator's licensed corpus and the model cache are mounted.
RUN useradd --system --uid 10001 --home /app newbieduo \
 && mkdir -p /data \
 && chown -R newbieduo:newbieduo /app /data
USER newbieduo

VOLUME ["/data"]
EXPOSE 8000

# Readiness is not "the port is open". The service answers hooks with 503 until
# the models are loaded and the five trigger answers are cached, so the health
# check asks for that state rather than for a TCP connection.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=20 \
  CMD curl -fsS "http://127.0.0.1:${NEWBIEDUO_PORT}/api/health" \
      | grep -q '"cds_state": *"ready"' || exit 1

ENTRYPOINT ["python", "scripts/demo_server.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--generate-timeout", "12"]
