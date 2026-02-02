import time
from contextlib import contextmanager


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"[func:{func.__name__}], {elapsed:.6f} secs")
        return result

    return wrapper


@contextmanager
def time_block(desc: str = None, enabled=True):
    if not enabled:
        yield
        return
    name = f"[{desc}] " if desc else ""

    print(f"[{name}] starts...")
    t0 = time.perf_counter()

    try:
        yield
    finally:
        t1 = time.perf_counter()
        print(f"[{name}] {t1 - t0:.4f} s")
