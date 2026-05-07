# ══════════════════════════════════════════════════════════════
# features.py — Rich hand feature extraction
#
# Replaces raw 63-float landmark dumps with a 151-float vector
# that encodes spatial geometry AND motion dynamics:
#
#   63  — wrist-relative, scale-normalised landmark coords
#   15  — joint angles (angle at each finger joint)
#   10  — fingertip pairwise distances  (C(5,2))
#   63  — frame-to-frame velocity of each landmark
#  ───
#  151  per hand  →  302 for two hands concatenated
# ══════════════════════════════════════════════════════════════

from __future__ import annotations
import numpy as np
from typing import Optional

# ── Joint triples (A, B, C) = angle at B ──────────────────────
# One triple per finger joint; 3 joints × 5 fingers = 15 angles.
_JOINT_TRIPLES = [
    (0, 1, 2), (1, 2, 3),  (2, 3, 4),          # Thumb
    (0, 5, 6), (5, 6, 7),  (6, 7, 8),           # Index
    (0, 9, 10),(9, 10, 11),(10, 11, 12),         # Middle
    (0,13, 14),(13,14, 15),(14, 15, 16),         # Ring
    (0,17, 18),(17,18, 19),(18, 19, 20),         # Pinky
]

_FINGERTIP_IDX = [4, 8, 12, 16, 20]             # thumb → pinky

# ── Feature-dimension constants (import these elsewhere) ───────
LANDMARK_DIM  = 63    # 21 landmarks × 3 coords
ANGLE_DIM     = 15
DISTANCE_DIM  = 10    # C(5,2) fingertip pairs
VELOCITY_DIM  = 63
FEATURE_DIM   = LANDMARK_DIM + ANGLE_DIM + DISTANCE_DIM + VELOCITY_DIM  # 151
FEATURE_DIM_2 = FEATURE_DIM * 2                                          # 302

EMPTY_HAND_FEATURES = np.zeros(FEATURE_DIM, dtype=np.float32)


# ── Helpers ───────────────────────────────────────────────────
def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle (radians) at joint B formed by A-B-C."""
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-7 or nc < 1e-7:
        return 0.0
    cos = np.dot(ba, bc) / (na * nc)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _normalise(hand_landmarks) -> np.ndarray:
    """
    Wrist-relative + max-abs scale normalisation.
    Returns (21, 3) float32 in approx [-1, 1].
    """
    lm    = hand_landmarks.landmark
    pts   = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
    pts  -= pts[0]                              # wrist → origin
    scale = np.max(np.abs(pts)) + 1e-6
    return pts / scale


# ── Public API ────────────────────────────────────────────────
def hand_to_features(
    hand_landmarks,
    prev_norm_flat: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a 151-float feature vector from one MediaPipe hand result.

    Args:
        hand_landmarks:  mediapipe NormalizedLandmarkList
        prev_norm_flat:  (63,) normalised coords from the previous frame,
                         used to compute velocity.  Pass None on the first frame.

    Returns:
        features   (151,) float32 — the full feature vector
        norm_flat  (63,)  float32 — normalised coords for the NEXT frame's velocity
    """
    norm      = _normalise(hand_landmarks)          # (21, 3)
    norm_flat = norm.flatten()                      # (63,)

    # 1. Landmark coords (63)
    coords = norm_flat.copy()

    # 2. Joint angles (15)
    angles = np.array(
        [_angle(norm[a], norm[b], norm[c]) for a, b, c in _JOINT_TRIPLES],
        dtype=np.float32,
    )

    # 3. Fingertip pairwise distances (10)
    tips = norm[_FINGERTIP_IDX]                     # (5, 3)
    dists = np.array(
        [np.linalg.norm(tips[i] - tips[j])
         for i in range(5) for j in range(i + 1, 5)],
        dtype=np.float32,
    )

    # 4. Frame-to-frame velocity (63)
    velocity = (norm_flat - prev_norm_flat).astype(np.float32) \
               if prev_norm_flat is not None \
               else np.zeros(LANDMARK_DIM, dtype=np.float32)

    features = np.concatenate([coords, angles, dists, velocity])
    return features.astype(np.float32), norm_flat


def mediapipe_to_norm63(hand_landmarks) -> np.ndarray:
    """
    Return (63,) normalised landmark vector only.
    Used by the legacy letter MLP which expects exactly 63 inputs.
    """
    return _normalise(hand_landmarks).flatten().astype(np.float32)
