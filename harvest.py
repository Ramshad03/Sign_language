# ==============================================================
# harvest.py — One-shot harvesting launcher
#
# Scans raw_videos/ and runs the correct pipeline for each folder:
#   • Single-char folders (A, B, C...) → harvest_letters.py → data/
#   • Word folders (hello, thanks...) → harvest_filter + extract → word_data/
#
# Usage:
#   python harvest.py              → process everything in raw_videos/
#   python harvest.py hello A bad  → process only those folders
#
# raw_videos/ folder structure:
#   raw_videos/
#     A/          ← letter: put .mp4 files here
#     B/
#     hello/      ← word: put .mp4 files here
#     thanks/
#
# After harvesting, run the training pipeline:
#   Letters:  python build_dataset.py → preprocess.py → train_model.py → evaluate_model.py
#   Words:    python word_dataset_builder.py → word_train.py
# ==============================================================

import os
import sys

RAW_VIDEO_DIR = "raw_videos"
_SPECIAL      = {"SPACE", "DEL", "NOTHING"}


def _classify_folders(names: list):
    """Split folder names into (letter_folders, word_folders)."""
    letters, words = [], []
    for n in names:
        if len(n) == 1 and n.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            letters.append(n.upper())
        elif n.upper() in _SPECIAL:
            letters.append(n.upper())
        else:
            words.append(n)
    return letters, words


def _all_folders():
    if not os.path.exists(RAW_VIDEO_DIR):
        return []
    return [
        d for d in os.listdir(RAW_VIDEO_DIR)
        if os.path.isdir(os.path.join(RAW_VIDEO_DIR, d))
    ]


def main():
    # ── Parse args ────────────────────────────────────────────
    requested = sys.argv[1:] if len(sys.argv) > 1 else None

    all_folders = _all_folders()
    if not all_folders:
        print(f"\n  ❌ raw_videos/ is empty or missing.")
        print("  Create subfolders for each gesture/word and place videos inside.\n")
        return

    folders = [f for f in all_folders if f in requested] if requested else all_folders

    if not folders:
        print(f"  ⚠️  None of {requested} found in raw_videos/")
        return

    letter_folders, word_folders = _classify_folders(folders)

    print("=" * 62)
    print("  HARVEST — auto-routing by folder type")
    print("=" * 62)
    print(f"  Letter folders : {letter_folders}")
    print(f"  Word folders   : {word_folders}")
    print("=" * 62)

    # ── Letters ───────────────────────────────────────────────
    if letter_folders:
        print("\n▶  Running harvest_letters.py ...\n")
        import harvest_letters
        harvest_letters.run(selected=letter_folders)

    # ── Words ─────────────────────────────────────────────────
    if word_folders:
        print("\n▶  Running harvest_filter.py ...\n")
        import harvest_filter
        harvest_filter.run_filter(selected=word_folders)

        print("\n▶  Running harvest_extract.py ...\n")
        import harvest_extract
        harvest_extract.run_extract(selected=word_folders)

    # ── Final instructions ────────────────────────────────────
    print("\n" + "=" * 62)
    print("  HARVESTING COMPLETE")
    print("=" * 62)

    if letter_folders:
        print("\n  Letter training pipeline:")
        print("    python build_dataset.py")
        print("    python preprocess.py")
        print("    python train_model.py")
        print("    python evaluate_model.py")

    if word_folders:
        print("\n  Word training pipeline:")
        print("    python word_dataset_builder.py")
        print("    python word_train.py")

    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
