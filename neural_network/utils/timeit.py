from time import perf_counter
from collections import defaultdict
from functools import wraps


class timeit:
    execution_times = defaultdict(list)

    groups = defaultdict(lambda: 0)

    @staticmethod
    def register(name: str, group_with: str):
        def decorator(func):
            @wraps(func)
            def inner(*args, **kwargs):
                nonlocal name
                start = perf_counter()
                result = func(*args, **kwargs)
                end = perf_counter()
                time = end - start
                timeit.execution_times[name].append(time)
                timeit.groups[group_with] += time
                return result
            return inner
        return decorator

    @staticmethod
    def get_data(name: str = None):
        if name:
            return timeit.execution_times[name]
        return dict(timeit.execution_times)

    @staticmethod
    def get_recent_execution_times(name: str, count: int = 1):
        if count == 1:
            return timeit.execution_times[name][-1]
        return timeit.execution_times[name][::-1][:count]

    @staticmethod
    def get_groups():
        return dict(timeit.groups)
