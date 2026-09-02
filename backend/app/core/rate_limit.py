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

    def allow(self, key: str, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        events = self._events[key]
        cutoff = current - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(current)
        return True

    def clear(self) -> None:
        """Clear state for deterministic tests and controlled application resets."""
        self._events.clear()
