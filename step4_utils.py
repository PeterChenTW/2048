"""Helpers for long, resumable training runs (step 4).

ReplayScheduler: fractional replay ratio (gradient updates per env step).
ActiveClock: counts only active training time; gaps longer than ``max_gap_s`` between ticks
(box freeze / suspend) are not counted toward the training budget.
atomic_save: torch.save to a temp file then os.replace, so a crash never leaves a torn checkpoint.
"""
import math
import os
import time


class ReplayScheduler:
    """Emit floor(ratio * steps) updates over time using a carried credit.

    ratio=0.78 -> 78 updates per 100 env steps, spread evenly. ratio may be > 1.
    """

    def __init__(self, ratio, credit=0.0):
        if isinstance(ratio, bool) or not math.isfinite(float(ratio)) or float(ratio) <= 0:
            raise ValueError('replay ratio must be finite and > 0')
        self.ratio = float(ratio)
        self.credit = float(credit)

    def step(self):
        """Call once per env step; returns number of updates to perform now."""
        self.credit += self.ratio
        n = int(math.floor(self.credit + 1e-9))
        self.credit -= n
        return n


class ActiveClock:
    def __init__(self, active_s=0.0, max_gap_s=60.0, now=time.monotonic):
        self.active_s = float(active_s)
        self.max_gap_s = float(max_gap_s)
        self.now = now
        self.last = now()
        self.skipped_s = 0.0

    def tick(self):
        t = self.now()
        dt = t - self.last
        self.last = t
        if 0 <= dt <= self.max_gap_s:
            self.active_s += dt
        else:
            self.skipped_s += max(dt, 0.0)
        return self.active_s


def atomic_save(obj, path):
    import torch
    path = str(path)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    with open(tmp, 'rb') as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
