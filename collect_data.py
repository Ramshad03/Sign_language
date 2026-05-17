# ─────────────────────────────────────────────────────────────────────────────
# collect_data.py
# Records 21 MediaPipe hand landmarks per gesture and saves them as .npy files
#
# Usage:
#   python collect_data.py            → collect all gestures
#   python collect_data.py X C SPACE  → collect only X, C, and SPACE
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import os
import sys
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR          = os.path.join(os.path.dirname(__file__), 'data')
SAMPLES_PER_CLASS = 100      # how many samples to collect per gesture
FRAME_SKIP        = 5        # capture 1 sample every N frames (adds variety)
CAMERA_INDEX      = 0        # 0 = default webcam

ALL_GESTURES = (
    list('ABCDEFHIKLMNORSTUVWXY') +['I_Love_You']+['BAD'] +  # J, Z, P, Q excluded — handled as motion/word sequences
    ['SPACE', 'DEL', 'NOTHING']
)

# ── Parse args:  [--clear] [GESTURE ...]  ─────────────────────────────────────
_args  = sys.argv[1:]
CLEAR  = "--clear" in _args
_args  = [a for a in _args if a != "--clear"]

if _args:
    requested = [a.upper() for a in _args]
    invalid   = [r for r in requested if r not in ALL_GESTURES]
    if invalid:
        print(f"  ⚠️  Unknown gestures ignored: {invalid}")
    GESTURES = [g for g in ALL_GESTURES if g in requested]
    print(f"  Collecting selected: {GESTURES}")
else:
    GESTURES = ALL_GESTURES

# ── Clear existing data if requested ─────────────────────────────────────────
if CLEAR:
    for g in GESTURES:
        g_dir = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), g)
        if os.path.exists(g_dir):
            removed = 0
            for f in os.listdir(g_dir):
                if f.endswith('.npy'):
                    os.remove(os.path.join(g_dir, f))
                    removed += 1
            print(f"  🗑️  Cleared {removed} samples from '{g}'")

# ── MEDIAPIPE SETUP ───────────────────────────────────────────────────────────

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw
import mediapipe.python.solutions.drawing_styles as mp_styles

# ── HELPERS ───────────────────────────────────────────────────────────────────

def extract_landmarks(hand_landmarks):
    """
    Flatten 21 hand landmarks into a (63,) float32 array.
    Each landmark has x, y, z → 21 × 3 = 63 values total.
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def create_dirs():
    """Create one sub-folder per gesture inside DATA_DIR."""
    for gesture in GESTURES:
        os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)


def draw_hud(frame, gesture, collected, total):
    """Draw progress bar and labels onto the webcam frame."""
    h, w = frame.shape[:2]

    # ── top label ─────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, f'Gesture: {gesture}',
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)

    # ── progress bar ──────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 20, h - 50, w - 40, 20
    filled = int((collected / total) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled,  bar_y + bar_h), (0, 220, 100), -1)
    cv2.putText(frame, f'{collected}/{total}',
                (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── hint ──────────────────────────────────────────────
    cv2.putText(frame, 'Q = skip this gesture',
                (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


def wait_for_ready(cap, hands, gesture):
    """
    Show a 'GET READY' screen and wait for SPACE.
    Returns False if user pressed Q to quit entirely.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # still run MediaPipe so user can see their hand
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, lms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        # overlay
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 60), -1)
        cv2.putText(frame, f'Next gesture: {gesture}',
                    (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 220, 255), 2)
        cv2.putText(frame, 'SPACE = start   Q = quit',
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.imshow('ASL Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False

# ── MAIN COLLECTION LOOP ─────────────────────────────────────────────────────

def collect():
    create_dirs()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.60,
    ) as hands:

        for gesture in GESTURES:

            # ── count existing samples (never skip — always add more) ──
            gesture_dir = os.path.join(DATA_DIR, gesture)
            existing    = len([f for f in os.listdir(gesture_dir) if f.endswith('.npy')])
            target      = existing + SAMPLES_PER_CLASS   # always collect a fresh batch

            print(f'\n{"─"*48}')
            print(f'  Gesture  : {gesture}')
            print(f'  Existing : {existing} samples')
            print(f'  Adding   : {SAMPLES_PER_CLASS} more  →  total {target}')
            print(f'{"─"*48}')

            # ── wait for user to get into position ────────
            ready = wait_for_ready(cap, hands, gesture)
            if not ready:
                break

            # ── 3-second countdown ────────────────────────
            for i in range(3, 0, -1):
                _, frame = cap.read()
                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, h), (20, 20, 20), -1)
                cv2.putText(frame, str(i),
                            (w // 2 - 30, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 200), 8)
                cv2.imshow('ASL Data Collection', frame)
                cv2.waitKey(1)
                time.sleep(1)

            # ── record samples — file index starts from existing ──────
            collected   = existing
            frame_count = 0

            while collected < target:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame     = cv2.flip(frame, 1)
                rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for lms in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, lms, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

                    # ── save every FRAME_SKIP-th frame ────
                    if frame_count % FRAME_SKIP == 0:
                        landmarks = extract_landmarks(results.multi_hand_landmarks[0])
                        save_path = os.path.join(gesture_dir, f'{collected}.npy')
                        np.save(save_path, landmarks)
                        collected   += 1

                frame_count += 1
                draw_hud(frame, gesture, collected - existing, SAMPLES_PER_CLASS)
                cv2.imshow('ASL Data Collection', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        # ── done ──────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()

    # ── summary ───────────────────────────────────────────
    print(f'\n{"═"*48}')
    print('  COLLECTION SUMMARY')
    print(f'{"═"*48}')
    for gesture in GESTURES:
        count  = len([f for f in os.listdir(os.path.join(DATA_DIR, gesture)) if f.endswith('.npy')])
        status = '✅' if count >= SAMPLES_PER_CLASS else f'⚠️  {count}/{SAMPLES_PER_CLASS}'
        print(f'  {gesture:8s} : {status}')
    print(f'{"═"*48}')


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    collect()