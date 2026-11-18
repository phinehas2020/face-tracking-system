"""Serialize PyTorch MPS work across cameras sharing the process Metal stream.

CoreML and CPU inference deliberately do not acquire this lock. Guard complete
operations, including tensor transfers back to CPU, not only model.forward().
"""

from contextlib import nullcontext
from functools import wraps
from threading import RLock

MPS_LOCK = RLock()
TRACKER_LOCK = RLock()  # Ultralytics track ID allocation is process-global.


def mps_guard(device):
    """A reentrant process-wide guard for a PyTorch MPS execution device."""
    return MPS_LOCK if str(device).split(":", 1)[0] == "mps" else nullcontext()


def serialized_mps(device_attribute):
    """Guard a synchronous instance method using its configured device field."""

    def decorate(method):
        @wraps(method)
        def guarded(self, *args, **kwargs):
            with mps_guard(getattr(self, device_attribute, "cpu")):
                return method(self, *args, **kwargs)

        return guarded

    return decorate
