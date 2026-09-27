"""Process-local SIGTERM handling; pass the cancellation predicate explicitly."""
from contextlib import contextmanager
import signal


@contextmanager
def termination_requested():
    requested = False

    def request(signum, frame):
        nonlocal requested
        requested = True

    previous = signal.signal(signal.SIGTERM, request)
    try:
        yield lambda: requested
    finally:
        signal.signal(signal.SIGTERM, previous)
