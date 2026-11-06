FROM python:3.11-slim

WORKDIR /app

# BLAS threads spin when several analyses run at once (measured ~10x CPU overhead);
# the analysis pool already provides parallelism.
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN useradd --system --uid 10001 --home-dir /app veritas \
    && mkdir /data && chown veritas /data

COPY --chown=veritas . /app

USER veritas
EXPOSE 8001

# One worker on purpose: rate limiters, result caches and the WebSocket
# registry are process-local. --proxy-headers makes request.client the real
# client (from nginx's X-Forwarded-For) so per-IP limits are not shared by
# everyone behind the proxy; this is safe because compose does not publish
# this port, so only nginx can reach it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--proxy-headers", "--forwarded-allow-ips", "*"]
