"profiler.py: profile pieces of code"

import timeit
from collections import UserDict
from contextlib import contextmanager
from threading import Lock

from numpy import std


class TagProfiler(UserDict[str, list[float]]):
    "tag based profiler"

    def __init__(self, dict: dict[str, list[float]] | None = None):
        super().__init__(dict)
        self._lock = Lock()

    @contextmanager
    def profile(self, key: str):
        "profile a piece of code and add it to a given key. Reentrant but not thread safe."
        try:
            start = timeit.default_timer()
            yield
            stop = timeit.default_timer()
            self.add(key, stop - start)
        finally:
            pass

    def add(self, key: str, value: float):
        with self._lock:
            self.setdefault(key, []).append(value)

    def cumulated_time(self):
        return sum((sum(t, 0.0) for t in self.values()), 0.0)

    def total_calls(self):
        return sum(map(len, self.values()), 0)

    def averages(self):
        for key, times in sorted(self.items()):
            num = len(times)
            yield key, sum(times) / num, num

    def stats(self):
        "iterator (key, mean, standard-deviation, number-of-sample)"
        for key, times in sorted(self.items()):
            num = len(times)
            mean = sum(times, 0.0) / num
            std_ = std(times, ddof=1)
            yield key, mean, std_, num
