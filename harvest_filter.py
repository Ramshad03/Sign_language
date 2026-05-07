# ==============================================================
# harvest_filter.py
# Reads every video in raw_videos/, runs MediaPipe on each frame,
# extracts 302-float rich features (coords + angles + distances +
# velocity) for BOTH hands, and saves to filtered/ as .npy files.
#
# Output shape per video: (N, 302)  — N valid frames
# ==============================================================

import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

from features import hand_to_features, EMPTY_HAND_FEATURES

# ── Config ────────────────────────────────────────────────────
RAW_VIDEO_DIR = "raw_videos"
FILTERED_DIR  = "filtered"

# Auto-detect word/gesture folders — any folder whose name is
# longer than 1 char or is a known multi-char token is a word.
_LETTER_SINGLES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"SPACE", "DEL", "NOTHING"}

def _detect_word_labels():
    if not os.path.exists(RAW_VIDEO_DIR):
        return []
    return sorted([
        d for d in os.listdir(RAW_VIDEO_DIR)
        if os.path.isdir(os.path.join(RAW_VIDEO_DIR, d))
        and d.upper() not in _LETTER_SINGLES
    ])

mp_hands = mp.solutions.hands


# ── Process one video → list of (302,) feature vectors ────────
def process_video(vid_path: str, hands) -> list:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return []

    frames_data = []
    prev_norm_h1 = None
    prev_norm_h2 = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)
        rgb.flags.writeable = True

        if res.multi_hand_landmarks:
            detected = res.multi_hand_landmarks

            feat_h1, norm_h1 = hand_to_features(detected[0], prev_norm_h1)
            prev_norm_h1 = norm_h1.copy()

            feat_h2 = EMPTY_HAND_FEATURES.copy()
            if len(detected) > 1:
                feat_h2, norm_h2 = hand_to_features(detected[1], prev_norm_h2)
                prev_norm_h2 = norm_h2.copy()
            else:
                prev_norm_h2 = None

            vec = np.concatenate([feat_h1, feat_h2])   # (302,)
            frames_data.append(vec)
        else:
            # Hand lost — reset velocity tracking
            prev_norm_h1 = None
            prev_norm_h2 = None

    cap.release()
    return frames_data


# ── Main ──────────────────────────────────────────────────────
def run_filter(selected: list = None) -> None:
    os.makedirs(FILTERED_DIR, exist_ok=True)

    total_saved   = 0
    total_skipped = 0

    print("=" * 62)
    print("  HARVEST FILTER — rich features (302 floats per frame)")
    print("=" * 62)

    all_labels = _detect_word_labels()
    if not all_labels:
        print("  ❌ No word folders found in raw_videos/")
        print("  Place word video folders like raw_videos/hello/, raw_videos/thanks/, etc.")
        return

    LABELS = [l for l in all_labels if l in selected] if selected else all_labels
    if not LABELS:
        print(f"  ⚠️  None of {selected} found in raw_videos/")
        return
    print(f"  Processing: {LABELS}\n")

    with mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    ) as hands:

        for label in LABELS:
            vid_folder  = os.path.join(RAW_VIDEO_DIR, label)
            save_folder = os.path.join(FILTERED_DIR,  label)
            os.makedirs(save_folder, exist_ok=True)

            if not os.path.exists(vid_folder):
                print(f"  [{label}] folder not found — skipping")
                continue

            videos = [f for f in os.listdir(vid_folder)
                      if f.lower().endswith((".mp4", ".avi", ".mov"))]

            if not videos:
                print(f"  [{label}] no videos found — skipping")
                continue

            print(f"\n  [{label}]")

            for vid_file in tqdm(videos, desc=f"    {label}", ncols=62):
                vid_path    = os.path.join(vid_folder, vid_file)
                frames_data = process_video(vid_path, hands)

                if len(frames_data) < 8:
                    print(f"    ⚠️  {vid_file} — only {len(frames_data)} "
                          f"good frames (need ≥8) — skipped")
                    total_skipped += 1
                    continue

                arr       = np.array(frames_data, dtype=np.float32)  # (N, 302)
                safe_name = (os.path.splitext(vid_file)[0]
                             .replace(" ", "_").replace(":", "-"))
                save_path = os.path.join(save_folder, f"{safe_name}.npy")
                np.save(save_path, arr)

                print(f"    ✅ {vid_file[:38]:<38} → {len(frames_data)} frames")
                total_saved += 1

    print("\n" + "=" * 62)
    print("  FILTER COMPLETE")
    print("=" * 62)
    print(f"  Saved   : {total_saved}")
    print(f"  Skipped : {total_skipped}")
    print(f"  Feature size per frame : 302  (2 × 151)")
    print(f"  Output  : {FILTERED_DIR}/")
    print("=" * 62)
    print("\n  Next: python harvest_extract.py\n")


if __name__ == "__main__":
    run_filter()
