# ══════════════════════════════════════════════════════════════
# collect_word_data.py — Segmenter-driven word data collection
#
# Usage:
#   python collect_word_data.py              → collect all words/letters
#   python collect_word_data.py hello Z bad  → collect only those three
#
# Workflow per word:
#   1. Word displayed — position your hand.
#   2. Press SPACE to arm the collector.
#   3. Sign the word naturally. The segmenter auto-captures.
#   4. Green border = captured. Repeat until quota reached.
#   5. Press Q to quit at any point.
#
# Output: word_data/<word>/<N>.npy  — shape (T, 302) variable T
# ══════════════════════════════════════════════════════════════

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time

from features  import (hand_to_features, EMPTY_HAND_FEATURES,
                        FEATURE_DIM_2, mediapipe_to_norm63)
from segmenter import GestureSegmenter, Phase

# ── Config ────────────────────────────────────────────────────
WORD_DATA_DIR     = "word_data"
SAMPLES_PER_WORD  = 60     # aim for more samples than the old 30-frame approach
CAMERA_INDEX      = 0

# ── Full vocabulary ───────────────────────────────────────────
# J and Z are motion letters — trained here as sequences, not
# in collect_data.py. app.py routes single-char results to
# add_letter() automatically so they appear correctly in text.
ALL_GESTURES = [
    "J", "Z", "P", "Q", "eat", "where", "wants","sleep", "food", "goodbye",
    "home","how",     # motion/ambiguous letters — routed to add_letter() automatically
    "hello", "thanks", "yes", "no", "please",
    "help",  "more",   "stop", "good", "bad",
    "sorry", "water",           # ← add new words here
]

# ── Parse args:  [--clear] [WORD ...]  ───────────────────────
_args  = sys.argv[1:]
CLEAR  = "--clear" in _args
_args  = [a for a in _args if a != "--clear"]

if _args:
    # Accept any word — new words don't need to be pre-added to ALL_GESTURES
    GESTURES = _args
    new_words = [w for w in GESTURES if w not in ALL_GESTURES]
    if new_words:
        print(f"  ✨ New words being added to vocabulary: {new_words}")
        ALL_GESTURES.extend(new_words)
    print(f"  Collecting: {GESTURES}")
else:
    GESTURES = ALL_GESTURES

# ── Clear existing data if requested ─────────────────────────
if CLEAR:
    for w in GESTURES:
        w_dir = os.path.join(WORD_DATA_DIR, w)
        if os.path.exists(w_dir):
            removed = 0
            for f in os.listdir(w_dir):
                if f.endswith('.npy'):
                    os.remove(os.path.join(w_dir, f))
                    removed += 1
            print(f"  🗑️  Cleared {removed} samples from '{w}'")


# ── HUD helpers ───────────────────────────────────────────────
def _draw_bar(frame, x, y, w, h, pct, color, label):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * pct), y + h), color, -1)
    cv2.putText(frame, label, (x + 4, y + h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)


_PHASE_COLORS = {
    "IDLE":    (80,  80,  80),
    "FORMING": (0,  140, 255),
    "STABLE":  (0,  200,  80),
}


def _draw_hud(frame, word, collected, total, seg: GestureSegmenter, armed: bool):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
    status = "ARMED — Sign now!" if armed else "Press SPACE to arm"
    col    = (0, 220, 100)  if armed else (180, 180, 180)
    cv2.putText(frame, f"'{word}'  {collected}/{total}", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    # Phase badge
    phase_col = _PHASE_COLORS.get(seg.phase_name, (80, 80, 80))
    cv2.rectangle(frame, (w - 150, 0), (w, 34), (25, 25, 25), -1)
    cv2.putText(frame, seg.phase_name, (w - 145, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_col, 2)

    # Buffer bar
    _draw_bar(frame, 10, h - 50, 300, 14,
              seg.buffer_progress(), phase_col,
              f"Buffer: {seg.buffer_len()} frames")

    # Progress bar
    pct = collected / total
    _draw_bar(frame, 10, h - 28, w - 20, 16, pct, (0, 200, 120),
              f"{collected}/{total} collected")


# ── Wait screen ───────────────────────────────────────────────
def _wait_for_space(cap, hands, word) -> bool:
    """Returns True when SPACE is pressed, False when Q is pressed."""
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)
        rgb.flags.writeable = True

        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 70), (15, 15, 50), -1)
        cv2.putText(frame, f"Next word: '{word}'", (14, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 220, 255), 2)
        cv2.putText(frame, "SPACE = start collection   Q = quit",
                    (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1)
        cv2.imshow("Word Collection v2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key == ord("q"):
            return False


# ── Main collection loop ──────────────────────────────────────
def collect() -> None:
    os.makedirs(WORD_DATA_DIR, exist_ok=True)
    for word in GESTURES:
        os.makedirs(os.path.join(WORD_DATA_DIR, word), exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    with mp_hands.Hands(
        static_image_mode       = False,
        max_num_hands           = 2,
        min_detection_confidence= 0.5,
        min_tracking_confidence = 0.5,
    ) as hands:

        for word in GESTURES:
            word_dir  = os.path.join(WORD_DATA_DIR, word)
            existing  = len([f for f in os.listdir(word_dir)
                             if f.endswith(".npy")])
            target    = existing + SAMPLES_PER_WORD   # always add a fresh batch

            print(f"\n{'─' * 52}")
            print(f"  Word     : '{word}'")
            print(f"  Existing : {existing} samples")
            print(f"  Adding   : {SAMPLES_PER_WORD} more  →  total {target}")
            print(f"{'─' * 52}")

            ready = _wait_for_space(cap, hands, word)
            if not ready:
                break

            # ── Per-word collection loop ───────────────────
            segmenter  = GestureSegmenter()
            prev_norm  = None
            armed      = True
            collected  = existing
            flash_end  = 0.0

            while collected < target:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                now   = time.time()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = hands.process(rgb)
                rgb.flags.writeable = True

                has_hand = bool(res.multi_hand_landmarks)

                if has_hand:
                    for lm in res.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, lm, mp_hands.HAND_CONNECTIONS,
                            mp_style.get_default_hand_landmarks_style(),
                            mp_style.get_default_hand_connections_style(),
                        )

                # ── Feature extraction ─────────────────────
                feat_h1 = EMPTY_HAND_FEATURES.copy()
                feat_h2 = EMPTY_HAND_FEATURES.copy()
                norm_h1 = np.zeros(63, dtype=np.float32)

                if has_hand:
                    feat_h1, norm_h1 = hand_to_features(
                        res.multi_hand_landmarks[0], prev_norm
                    )
                    prev_norm = norm_h1.copy()
                    if len(res.multi_hand_landmarks) > 1:
                        feat_h2, _ = hand_to_features(
                            res.multi_hand_landmarks[1], None
                        )
                else:
                    prev_norm = None

                feat_combined = np.concatenate([feat_h1, feat_h2])

                # ── Segmenter update ───────────────────────
                segment = None
                if armed:
                    segment, _ = segmenter.update(feat_combined, norm_h1, has_hand)

                # ── Save completed segment ─────────────────
                if segment:
                    arr       = np.array(segment, dtype=np.float32)
                    save_path = os.path.join(word_dir, f"{collected}.npy")
                    np.save(save_path, arr)
                    collected += 1
                    flash_end  = now + 0.4
                    print(f"  ✓ Captured  shape={arr.shape}  "
                          f"[{collected - existing}/{SAMPLES_PER_WORD}]")

                # ── HUD ────────────────────────────────────
                _draw_hud(frame, word, collected - existing, SAMPLES_PER_WORD,
                          segmenter, armed)

                # Green flash on capture
                if now < flash_end:
                    h_f, w_f = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (w_f - 1, h_f - 1),
                                  (0, 255, 100), 8)
                    cv2.putText(frame, "CAPTURED!", (w_f // 2 - 100, h_f // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 100), 4)

                cv2.imshow("Word Collection v2", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord(" "):
                    armed = not armed
                    if armed:
                        segmenter.reset()
                        prev_norm = None
                    print(f"  {'Armed' if armed else 'Paused'}")
                elif key == ord("r"):
                    # Re-arm + reset segmenter
                    segmenter.reset()
                    prev_norm = None
                    armed     = True
                    print("  Segmenter reset, re-armed")

    # ── Summary ───────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'═' * 52}")
    print("  COLLECTION SUMMARY")
    print(f"{'═' * 52}")
    for word in GESTURES:
        count = len([f for f in os.listdir(os.path.join(WORD_DATA_DIR, word))
                     if f.endswith(".npy")])
        ok    = "✅" if count >= SAMPLES_PER_WORD else f"⚠️  {count}/{SAMPLES_PER_WORD}"
        print(f"  {word:<20} {ok}")
    print(f"{'═' * 52}")
    print("\n  Next step: python word_dataset_builder.py\n")


if __name__ == "__main__":
    collect()
