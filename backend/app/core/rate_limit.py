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
