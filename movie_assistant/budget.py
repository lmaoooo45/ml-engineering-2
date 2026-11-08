from time import monotonic


class RequestBudget:
    def __init__(self, seconds=180, max_calls=12):
        self.deadline = monotonic() + seconds
        self.max_calls = max_calls
        self.calls = 0

    def remaining(self):
        left = self.deadline - monotonic()
        if left <= 0:
            raise TimeoutError("Истёк общий бюджет запроса: 180 секунд")
        return left

    def before_call(self, endpoint):
        self.remaining()
        if endpoint == "/api/chat":
            if self.calls >= self.max_calls:
                raise RuntimeError("Достигнут предел генеративных вызовов")
            self.calls += 1
