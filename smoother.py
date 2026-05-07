# ══════════════════════════════════════════════════════════════
# smoother.py — Temporal confidence accumulation
#
# Replaces the hard majority-vote buffer with an exponential
# moving average (EMA) over raw softmax probability vectors.
#
# Benefits vs. majority vote:
#   • Soft weighting — recent frames matter more than old ones
#   • Trend detection — can tell if confidence is rising or falling
#   • No discrete "window" artefact when the dominant class changes
# ══════════════════════════════════════════════════════════════

from __future__ import annotations
import numpy as np
from collections import deque


class ConfidenceSmoother:
    """
    Exponential moving average over class probability distributions.

    Args:
        num_classes:  Number of output classes.
        alpha:        EMA weight for the newest frame (0 < alpha < 1).
                      Higher → faster response; lower → smoother but laggier.
        history:      Window size kept for trend computation.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.35,
        history: int = 12,
    ) -> None:
        self.alpha       = alpha
        self.num_classes = num_classes
        self._ema        = np.zeros(num_classes, dtype=np.float64)
        self._peak_hist  = deque(maxlen=history)
        self._ready      = False

    # ── Update ────────────────────────────────────────────────
    def update(self, probs: np.ndarray) -> np.ndarray:
        """
        Feed a softmax probability vector for the current frame.
        Returns the smoothed distribution as float32.
        """
        p = probs.astype(np.float64)
        if not self._ready:
            self._ema   = p.copy()
            self._ready = True
        else:
            self._ema = self.alpha * p + (1.0 - self.alpha) * self._ema

        self._peak_hist.append(float(self._ema.max()))
        return self._ema.astype(np.float32)

    # ── Read ──────────────────────────────────────────────────
    def best(self) -> tuple[int, float]:
        """Returns (class_index, smoothed_confidence) for the top class."""
        idx = int(self._ema.argmax())
        return idx, float(self._ema[idx])

    def trend(self) -> float:
        """
        Linear slope of peak confidence over the stored history.
        Positive = rising (model is getting more certain).
        Negative = falling (model is becoming uncertain → suppress output).
        Returns 0.0 if fewer than 3 samples have been seen.
        """
        h = list(self._peak_hist)
        if len(h) < 3:
            return 0.0
        x = np.arange(len(h), dtype=np.float32)
        slope, _ = np.polyfit(x, h, 1)
        return float(slope)

    # ── Reset ─────────────────────────────────────────────────
    def reset(self) -> None:
        self._ema[:]  = 0.0
        self._peak_hist.clear()
        self._ready   = False
