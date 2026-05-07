# ══════════════════════════════════════════════════════════════
# segmenter.py — Motion-driven gesture segmentation
#
# Replaces the fixed 30-frame sliding window with a state machine
# that buffers only the meaningful part of a gesture:
#
#   IDLE     → no motion; models are suppressed
#   FORMING  → motion detected; frames are buffered
#   STABLE   → velocity has settled; letter model may fire here
#   COMPLETE → gesture ended; word model receives the segment
#
# Transitions:
#   IDLE    → FORMING  : landmark velocity > MOTION_THRESH
#   FORMING → STABLE   : velocity < SETTLE_THRESH for SETTLE_FRAMES
#   STABLE  → FORMING  : new motion burst (emits current segment first)
#   any     → IDLE     : hand absent for IDLE_GAP_MAX frames,
#                        or buffer reaches MAX_FRAMES
# ══════════════════════════════════════════════════════════════

from __future__ import annotations
from enum import Enum, auto
import numpy as np


class Phase(Enum):
    IDLE     = auto()
    FORMING  = auto()
    STABLE   = auto()


class GestureSegmenter:
    # ── Tunable thresholds ─────────────────────────────────────
    # Velocity is L2-norm over 63 coordinates. Natural hand tremor
    # while holding a letter creates ~0.020–0.060, so thresholds
    # must be set well above that to avoid staying stuck in FORMING.
    MOTION_THRESH  = 0.120   # deliberate inter-letter movement
    SETTLE_THRESH  = 0.090   # "holding still" — allows natural tremor incl. bent fingers
    SETTLE_FRAMES  = 5       # consecutive low-velocity frames to confirm STABLE
    MIN_FRAMES     = 8       # discard segments shorter than this
    MAX_FRAMES     = 90      # force-emit when buffer reaches this length
    IDLE_GAP_MAX   = 10      # frames without a hand before full reset

    def __init__(self) -> None:
        self.phase          = Phase.IDLE
        self._buffer: list  = []
        self._settle_cnt    = 0
        self._idle_cnt      = 0
        self._prev_norm     = None   # (63,) landmark coords from last frame
        self._stable_seen   = False  # True if STABLE was reached in this segment

    # ── Read-only properties ───────────────────────────────────
    @property
    def phase_name(self) -> str:
        return self.phase.name

    def buffer_len(self) -> int:
        return len(self._buffer)

    def buffer_progress(self) -> float:
        """0.0 → 1.0 fill level relative to MAX_FRAMES."""
        return len(self._buffer) / self.MAX_FRAMES

    # ── Main update (call every frame) ────────────────────────
    def update(
        self,
        feature_vec: np.ndarray,
        norm_63: np.ndarray,
        has_hand: bool,
    ):
        """
        Feed one frame.

        Args:
            feature_vec: (302,) rich feature vector for this frame
            norm_63:     (63,) normalised landmark coords used for velocity
            has_hand:    whether MediaPipe detected any hand

        Returns:
            (segment, was_stable) where segment is a list of feature vectors
            if a gesture just finished, else (None, False).
            was_stable=True means the hand held a letter position at some
            point — the segment should NOT be sent to the word model.
        """
        if not has_hand:
            return self._handle_no_hand()

        self._idle_cnt = 0

        # Velocity = L2 distance between consecutive normalised coord vectors
        vel = float(np.linalg.norm(norm_63 - self._prev_norm)) \
              if self._prev_norm is not None else 0.0
        self._prev_norm = norm_63.copy()

        return self._transition(feature_vec, vel)

    # ── State transitions ─────────────────────────────────────
    def _handle_no_hand(self):
        self._idle_cnt += 1
        if self._idle_cnt >= self.IDLE_GAP_MAX:
            seg, was_stable = self._maybe_emit()
            self.reset()
            return seg, was_stable
        return None, False

    def _transition(self, feat: np.ndarray, vel: float):
        if self.phase == Phase.IDLE:
            if vel > self.MOTION_THRESH:
                self.phase        = Phase.FORMING
                self._buffer      = [feat]
                self._settle_cnt  = 0
                self._stable_seen = False

        elif self.phase == Phase.FORMING:
            self._buffer.append(feat)
            if vel < self.SETTLE_THRESH:
                self._settle_cnt += 1
                if self._settle_cnt >= self.SETTLE_FRAMES:
                    self.phase        = Phase.STABLE
                    self._stable_seen = True       # hand settled = letter position
            else:
                self._settle_cnt = 0
            if len(self._buffer) >= self.MAX_FRAMES:
                return self._force_emit()

        elif self.phase == Phase.STABLE:
            self._buffer.append(feat)
            if vel > self.MOTION_THRESH:
                seg, was_stable = self._maybe_emit()
                self.phase        = Phase.FORMING
                self._buffer      = [feat]
                self._settle_cnt  = 0
                self._stable_seen = False
                self._prev_norm   = None
                return seg, was_stable
            if len(self._buffer) >= self.MAX_FRAMES:
                return self._force_emit()

        return None, False

    # ── Emit helpers ──────────────────────────────────────────
    def _maybe_emit(self):
        if len(self._buffer) >= self.MIN_FRAMES:
            return list(self._buffer), self._stable_seen
        return None, self._stable_seen

    def _force_emit(self):
        seg, was_stable = self._maybe_emit()
        self.reset()
        return seg, was_stable

    # ── Public reset ──────────────────────────────────────────
    def reset(self) -> None:
        self.phase        = Phase.IDLE
        self._buffer      = []
        self._settle_cnt  = 0
        self._idle_cnt    = 0
        self._prev_norm   = None
        self._stable_seen = False
