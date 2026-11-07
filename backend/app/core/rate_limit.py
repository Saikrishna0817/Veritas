"""Small, dependency-free rate limiting for narrowly scoped abuse controls."""

from __future__ import annotations

from collections import defaultdict, deque
from time import monotonic


class SlidingWindowRateLimiter:
    """Allow a fixed number of events per key inside a rolling time window."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._calls = 0

    def allow(self, key: str, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        events = self._events[key]
        cutoff = current - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(current)
        self._calls += 1
        if self._calls % 1000 == 0:
            self._prune(current)
        return True

    def _prune(self, now: float) -> None:
        """Drop keys with no events left in the window so the dict can't grow forever."""
        cutoff = now - self.window_seconds
        for key in [k for k, ev in self._events.items() if not ev or ev[-1] <= cutoff]:
            del self._events[key]

    def clear(self) -> None:
        """Clear state for deterministic tests and controlled application resets."""
        self._events.clear()


# ── Shared (multi-process) limiter ───────────────────────────────────────────────────────────────

_LUA = """
local key, now, window, limit, member = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
if redis.call('ZCARD', key) >= limit then return 0 end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, math.ceil(window * 1000))
return 1
"""


class RedisRateLimiter:
    """Same interface as SlidingWindowRateLimiter, but the window is shared by every API process through Redis.

    If Redis is unreachable it falls back to a per-process limiter (and logs), so an outage degrades protection
    instead of taking the API down."""

    def __init__(self, url: str, name: str, limit: int, window_seconds: float) -> None:
        import redis
        self.name, self.limit, self.window_seconds = name, limit, window_seconds
        self._redis = redis.Redis.from_url(url, socket_timeout=1, socket_connect_timeout=1)
        self._script = self._redis.register_script(_LUA)
        self._local = SlidingWindowRateLimiter(limit, window_seconds)

    def allow(self, key: str, now: float | None = None) -> bool:
        import logging
        import time
        import uuid
        try:
            current = time.time() if now is None else now
            return bool(self._script(keys=[f"rl:{self.name}:{key}"], args=[current, self.window_seconds, self.limit, uuid.uuid4().hex]))
        except Exception:
            logging.getLogger(__name__).warning("Redis rate limiter unavailable; using the per-process limiter", exc_info=True)
            return self._local.allow(key, now)

    def clear(self) -> None:
        try:
            for k in self._redis.scan_iter(f"rl:{self.name}:*"):
                self._redis.delete(k)
        except Exception:
            pass
        self._local.clear()

    @property
    def _events(self):          # tests clear limiter state through this on the in-memory class
        return _ClearProxy(self)


class _ClearProxy:
    def __init__(self, limiter): self._limiter = limiter
    def clear(self): self._limiter.clear()


def make_limiter(name: str, limit: int, window_seconds: float):
    """Redis-backed when VERITAS_REDIS_URL is set, otherwise in-process."""
    from app.core.config import settings
    if settings.redis_url:
        return RedisRateLimiter(settings.redis_url, name, limit, window_seconds)
    return SlidingWindowRateLimiter(limit, window_seconds)
