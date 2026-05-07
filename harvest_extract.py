# ==============================================================
# harvest_extract.py
# Reads filtered (N, 302) arrays from filtered/ and uses the
# GestureSegmenter to extract variable-length gesture segments.
# Each saved file has shape (T, 302) where T is the natural
# duration of the detected gesture in that video.
# ==============================================================

import numpy as np
import os
import sys
from tqdm import tqdm

from segmenter import GestureSegmenter

# ── Config ────────────────────────────────────────────────────
FILTERED_DIR  = "filtered"
WORD_DATA_DIR = "word_data"


def _detect_labels():
    if not os.path.exists(FILTERED_DIR):
        return []
    return sorted([
        d for d in os.listdir(FILTERED_DIR)
        if os.path.isdir(os.path.join(FILTERED_DIR, d))
    ])


# ── Extract variable-length segments using the segmenter ──────
def extract_segments(frames: np.ndarray) -> list:
    segmenter = GestureSegmenter()
    segments  = []

    for feat_vec in frames:
        norm_63          = feat_vec[:63].copy()
        segment, _       = segmenter.update(feat_vec, norm_63, has_hand=True)  # unpack tuple
        if segment:
            segments.append(np.array(segment, dtype=np.float32))

    # Flush remaining buffer at end of video
    if segmenter.buffer_len() >= GestureSegmenter.MIN_FRAMES:
        segments.append(np.array(segmenter._buffer, dtype=np.float32))

    return segments


# ── Main ──────────────────────────────────────────────────────
def run_extract(selected: list = None) -> None:
    os.makedirs(WORD_DATA_DIR, exist_ok=True)

    all_labels = _detect_labels()
    if not all_labels:
        print("  ❌ No folders found in filtered/ — run harvest_filter.py first.")
        return

    # Filter to selected labels if specified
    LABELS = [l for l in all_labels if l in selected] if selected else all_labels

    if not LABELS:
        print(f"  ⚠️  None of {selected} found in filtered/")
        return

    print("=" * 62)
    print("  HARVEST EXTRACT — segmenter-driven variable-length")
    print("=" * 62)
    print(f"  Processing: {LABELS}\n")

    total_segments = 0
    report         = []

    for label in LABELS:
        filt_folder = os.path.join(FILTERED_DIR,  label)
        save_folder = os.path.join(WORD_DATA_DIR, label)
        os.makedirs(save_folder, exist_ok=True)

        if not os.path.exists(filt_folder):
            print(f"  [{label}] filtered folder not found — skipping")
            continue

        npy_files = [f for f in os.listdir(filt_folder) if f.endswith(".npy")]
        if not npy_files:
            print(f"  [{label}] no filtered files found — skipping")
            continue

        existing    = len([f for f in os.listdir(save_folder) if f.endswith(".npy")])
        seq_idx     = existing
        label_count = 0

        for npy_file in tqdm(npy_files, desc=f"  {label:<14}", ncols=62):
            frames = np.load(os.path.join(filt_folder, npy_file))

            if frames.ndim != 2 or frames.shape[1] != 302:
                continue

            for seg in extract_segments(frames):
                np.save(os.path.join(save_folder, f"{seq_idx}.npy"), seg)
                seq_idx     += 1
                label_count += 1

        total_segments += label_count
        report.append((label, label_count))

    print("\n" + "=" * 62)
    print("  EXTRACT COMPLETE")
    print("=" * 62)
    print(f"  {'Label':<20} {'Segments':>10}")
    print("  " + "─" * 34)
    for label, count in report:
        flag = "✅" if count >= 10 else "⚠️ "
        print(f"  {label:<20} {count:>10}   {flag}")
    print(f"  {'TOTAL':<20} {total_segments:>10}")
    print("=" * 62)
    print(f"\n  Next: python word_dataset_builder.py\n")


if __name__ == "__main__":
    sel = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else None
    run_extract(selected=sel)
