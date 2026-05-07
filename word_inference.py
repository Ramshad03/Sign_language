# ══════════════════════════════════════════════════════════════
# word_inference.py — Word inference wrapper
#
# The new system uses GestureTransformer (gesture_model.py) with
# event-based segmentation instead of a sliding-window LSTM.
# This module wraps that for standalone use or testing.
# ══════════════════════════════════════════════════════════════

from __future__ import annotations
import numpy as np
import torch
import os

from gesture_model import load_checkpoint
from features      import FEATURE_DIM_2

MODEL_PATH = os.path.join("model", "word_model.pt")
MAX_LEN    = 90


class WordInferencer:
    """
    Run word inference on a complete gesture segment.

    Usage:
        inf = WordInferencer()
        label, conf = inf.predict(segment)   # segment = list of (302,) arrays
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None
        self.labels = []

        if os.path.exists(MODEL_PATH):
            self.model, self.labels = load_checkpoint(MODEL_PATH, self.device)
            print(f"  [WordInferencer] {len(self.labels)} classes | {self.device}")
        else:
            print("  [WordInferencer] model not found — predict() will return None")

    def predict(self, segment: list) -> tuple[str | None, float]:
        """
        Args:
            segment: list of (302,) float32 feature vectors — one per frame.

        Returns:
            (label, confidence) or (None, 0.0) if model not loaded.
        """
        if self.model is None or not segment:
            return None, 0.0

        T   = min(len(segment), MAX_LEN)
        arr = np.zeros((MAX_LEN, FEATURE_DIM_2), dtype=np.float32)
        for i, feat in enumerate(segment[:T]):
            arr[i] = feat

        mask     = np.ones(MAX_LEN, dtype=bool)
        mask[:T] = False

        x_t = torch.tensor(arr).unsqueeze(0).to(self.device)
        m_t = torch.tensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(x_t, m_t), dim=1)[0].cpu().numpy()

        idx = int(probs.argmax())
        return self.labels[idx], float(probs[idx])
