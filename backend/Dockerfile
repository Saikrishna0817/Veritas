FROM python:3.11-slim

WORKDIR /app

# BLAS threads spin when several analyses run at once (measured ~10x CPU overhead);
# the analysis pool already provides parallelism.
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN useradd --system --uid 10001 --home-dir /app veritas \
    && mkdir /data /uploads && chown veritas /data /uploads

COPY --chown=veritas . /app

USER veritas
EXPOSE 8001

# One API process on purpose: the WebSocket registry and a few small caches are process-local. Scale analysis by
# running more `worker` containers (python -m app.jobs.worker), not more API workers. --proxy-headers makes
# request.client the real client (from nginx's X-Forwarded-For); safe because compose does not publish this port.
# Migrations run first; they are idempotent, so restarts and rolling deploys are fine.
CMD ["sh", "-c", "python -m app.models.migrate && exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --proxy-headers --forwarded-allow-ips '*'"]
