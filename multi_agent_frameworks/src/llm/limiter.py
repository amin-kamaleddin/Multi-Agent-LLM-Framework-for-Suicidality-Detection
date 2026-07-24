"""Shared concurrency guard for model calls made by one evaluation run."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Iterator


class LlmConcurrencyLimiter:
    """Bound the number of simultaneous model requests across all case workers."""

    def __init__(self, max_in_flight: int):
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be greater than zero")
        self.max_in_flight = int(max_in_flight)
        self._semaphore = threading.BoundedSemaphore(self.max_in_flight)
        self._lock = threading.Lock()
        self._active = 0
        self._max_observed = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        self._semaphore.acquire()
        with self._lock:
            self._active += 1
            self._max_observed = max(self._max_observed, self._active)
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    @property
    def max_observed(self) -> int:
        with self._lock:
            return self._max_observed

    def wrap(self, model: Any) -> "ConcurrencyLimitedModel":
        return ConcurrencyLimitedModel(model=model, limiter=self)


class ConcurrencyLimitedModel:
    """Transparent model proxy that acquires a run-wide request slot."""

    def __init__(self, *, model: Any, limiter: LlmConcurrencyLimiter):
        self._model = model
        self._limiter = limiter

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        with self._limiter.slot():
            return self._model.invoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
