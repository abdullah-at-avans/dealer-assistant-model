import time
from .printer import log_info

def timed(label, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    log_info(f"{label}: {end - start:.3f}s")

    return result