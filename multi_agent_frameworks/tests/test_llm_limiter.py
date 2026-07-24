from concurrent.futures import ThreadPoolExecutor
import threading
import time

from llm.limiter import LlmConcurrencyLimiter


class TrackingModel:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def invoke(self, messages):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)
            return messages
        finally:
            with self.lock:
                self.active -= 1


def test_limiter_bounds_requests_across_shared_model_wrappers():
    model = TrackingModel()
    limiter = LlmConcurrencyLimiter(3)
    wrappers = [limiter.wrap(model) for _ in range(5)]

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(wrappers[index % len(wrappers)].invoke, [{"content": index}])
            for index in range(30)
        ]
        assert [future.result() for future in futures]

    assert model.max_active == 3
    assert limiter.max_observed == 3
