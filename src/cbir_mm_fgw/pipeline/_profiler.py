from ..profiler import TagProfiler

_profiler = TagProfiler()


def get_times():
    "deep copy of the profiler state"
    return {k: v[:] for k, v in _profiler.items()}


def clear_profiler():
    _profiler.clear()
