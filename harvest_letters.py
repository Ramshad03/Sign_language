# ==============================================================
# harvest_letters.py
# Extracts letter training data from pre-recorded videos.
#
# Folder structure expected:
#   raw_videos/
#     A/  video1.mp4  video2.mp4 ...
#     B/  video1.mp4 ...
#     SPACE/ ...
#     DEL/   ...
#
# Output:  data/<LETTER>/<N>.npy  — each file is one (63,) snapshot
#
# Usage:
#   python harvest_letters.py            → process all letter folders
#   python harvest_letters.py X C SPACE  → process only X, C, SPACE
# ==============================================================

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────
RAW_VIDEO_DIR = "raw_videos"
DATA_DIR      = "data"
FRAME_SKIP    = 5    # save 1 snapshot every N frames (adds variety)

# ── Auto-detect letter labels from raw_videos/ ────────────────
# A folder is treated as a LETTER if its name is a single uppercase
# character A-Z or one of the special tokens.
_SPECIAL = {"SPACE", "DEL", "NOTHING"}

def _is_letter_folder(name: str) -> bool:
    return (len(name) == 1 and name.isupper()) or name in _SPECIAL

def _detect_labels():
    if not os.path.exists(RAW_VIDEO_DIR):
        return []
    return sorted([
        d for d in os.listdir(RAW_VIDEO_DIR)
        if os.path.isdir(os.path.join(RAW_VIDEO_DIR, d))
        and _is_letter_folder(d)
    ])

# ── Normalisation (matches collect_data.py exactly) ───────────
def _normalise(hand_landmarks) -> np.ndarray:
    lm    = hand_landmarks.landmark
    base  = np.array([lm[0].x, lm[0].y, lm[0].z])
    pts   = np.array([[l.x, l.y, l.z] for l in lm]) - base
    scale = np.max(np.abs(pts)) + 1e-6
    return (pts / scale).flatten().astype(np.float32)

# ── Extract snapshots from one video ─────────────────────────
def _extract_snapshots(vid_path: str, hands) -> list:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return []

    snapshots  = []
    frame_idx  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            if res.multi_hand_landmarks:
                snapshots.append(_normalise(res.multi_hand_landmarks[0]))

        frame_idx += 1

    cap.release()
    return snapshots

# ── Main ──────────────────────────────────────────────────────
def run(selected: list = None) -> None:
    all_labels = _detect_labels()

    if not all_labels:
        print(f"\n  ❌ No letter folders found in {RAW_VIDEO_DIR}/")
        print("  Create folders like raw_videos/A/, raw_videos/B/, etc.")
        print("  and place .mp4 / .avi / .mov files inside them.\n")
        return

    labels = [l for l in all_labels if l in selected] if selected else all_labels

    if not labels:
        print(f"  ⚠️  None of {selected} found in {RAW_VIDEO_DIR}/ — nothing to do.")
        return

    print("=" * 62)
    print("  HARVEST LETTERS")
    print("=" * 62)
    print(f"  Labels found : {all_labels}")
    print(f"  Processing   : {labels}")
    print("=" * 62)

    mp_hands = mp.solutions.hands
    total_saved = 0
    report = []

    with mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = 0.75,
        min_tracking_confidence  = 0.60,
    ) as hands:

        for label in labels:
            vid_folder  = os.path.join(RAW_VIDEO_DIR, label)
            save_folder = os.path.join(DATA_DIR, label)
            os.makedirs(save_folder, exist_ok=True)

            videos = [f for f in os.listdir(vid_folder)
                      if f.lower().endswith((".mp4", ".avi", ".mov"))]

            if not videos:
                print(f"\n  [{label}] no videos found — skipping")
                continue

            # File counter starts from existing count (never overwrites)
            existing = len([f for f in os.listdir(save_folder)
                            if f.endswith(".npy")])
            idx          = existing
            label_count  = 0

            print(f"\n  [{label}]  ({existing} snapshots already exist)")

            for vid_file in tqdm(videos, desc=f"    {label}", ncols=62):
                vid_path  = os.path.join(vid_folder, vid_file)
                snaps     = _extract_snapshots(vid_path, hands)

                if not snaps:
                    print(f"    ⚠️  {vid_file} — no hand detected, skipped")
                    continue

                for snap in snaps:
                    np.save(os.path.join(save_folder, f"{idx}.npy"), snap)
                    idx         += 1
                    label_count += 1

                print(f"    ✅ {vid_file[:42]:<42} → {len(snaps)} snapshots")

            total_saved  += label_count
            report.append((label, existing, label_count, idx))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  HARVEST LETTERS — COMPLETE")
    print("=" * 62)
    print(f"  {'Label':<10} {'Before':>8}  {'Added':>8}  {'Total':>8}")
    print("  " + "─" * 40)
    for lbl, before, added, total in report:
        print(f"  {lbl:<10} {before:>8}  {added:>8}  {total:>8}")
    print("=" * 62)
    print(f"  Snapshots saved this run : {total_saved}")
    print(f"  Output folder            : {DATA_DIR}/")
    print("\n  Next steps:")
    print("    python build_dataset.py")
    print("    python preprocess.py")
    print("    python train_model.py")
    print("    python evaluate_model.py")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    selected = [a.upper() for a in args] if args else None
    run(selected)
