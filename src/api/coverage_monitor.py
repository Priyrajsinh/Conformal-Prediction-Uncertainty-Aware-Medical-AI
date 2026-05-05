"""Rolling coverage drift monitor with Prometheus instrumentation.

Exposes two metrics (rules C5, C10, C42):
- ``coverage_violations_total`` (Counter) — fired when empirical coverage
  drops below ``1 - alpha - epsilon`` over the last *window_size* labelled samples.
- ``mean_set_size`` (Gauge) — rolling mean prediction-set size over all
  recent requests (labelled and unlabelled).
"""

from collections import deque
from threading import Lock
from typing import Optional

from prometheus_client import Counter, Gauge

COVERAGE_VIOLATIONS: Counter = Counter(
    "coverage_violations_total",
    "Times observed coverage dropped below 1-alpha-epsilon",
)
MEAN_SET_SIZE_GAUGE: Gauge = Gauge(
    "mean_set_size",
    "Rolling mean prediction set size",
)


class CoverageMonitor:
    """Thread-safe rolling window monitor for prediction-set coverage drift."""

    def __init__(self, window_size: int, epsilon: float) -> None:
        """Initialise with a fixed rolling *window_size* and *epsilon* slack."""
        self.window: deque[dict] = deque(maxlen=window_size)
        self.epsilon = epsilon
        self.lock = Lock()

    def record(
        self,
        pred_set: list[int],
        alpha: float,
        true_label: Optional[int] = None,
    ) -> None:
        """Record a new prediction and optionally check coverage against the guarantee.

        *true_label* is None for live inference and supplied when labelled
        feedback arrives (Day 8 hook). Coverage is only checked when at least
        50 labelled samples have accumulated.
        """
        with self.lock:
            self.window.append({"set": pred_set, "alpha": alpha, "y_true": true_label})
            sizes = [len(x["set"]) for x in self.window]
            if sizes:
                MEAN_SET_SIZE_GAUGE.set(sum(sizes) / len(sizes))
            labelled = [x for x in self.window if x["y_true"] is not None]
            if len(labelled) >= 50:
                covered = sum(x["y_true"] in x["set"] for x in labelled) / len(labelled)
                a = labelled[0]["alpha"]
                if covered < (1.0 - a - self.epsilon):
                    COVERAGE_VIOLATIONS.inc()
