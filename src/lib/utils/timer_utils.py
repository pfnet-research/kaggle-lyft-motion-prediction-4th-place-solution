from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer(name):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print("[{}] done in {:.3f} s".format(name, t1 - t0))


@contextmanager
def timer_ms(name):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print("[{}] done in {:.3f} ms".format(name, (t1 - t0) * 1000.))
