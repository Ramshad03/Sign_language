# ══════════════════════════════════════════════════════════════
# word_dataset_builder.py — Variable-length gesture dataset with augmentation
# ══════════════════════════════════════════════════════════════

import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────
WORD_DATA_DIR = "word_data"
MODEL_DIR     = "model"
SAVE_PATH     = os.path.join(MODEL_DIR, "word_dataset.npz")
LABELS_PATH   = os.path.join(MODEL_DIR, "word_labels.json")

MAX_LEN       = 90
MIN_LEN       = 8
INPUT_SIZE    = 302
TEST_SIZE     = 0.20
RANDOM_SEED   = 42
AUG_COPIES    = 3


# ── Temporal augmentation ─────────────────────────────────────
def _aug_noise(seq: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, 0.006, seq.shape).astype(np.float32)
    return np.clip(seq + noise, -2.0, 2.0)

def _aug_speedup(seq: np.ndarray) -> np.ndarray:
    T   = len(seq)
    n   = max(MIN_LEN, int(T * 0.75))
    idx = np.linspace(0, T - 1, n, dtype=int)
    return seq[idx]

def _aug_slowdown(seq: np.ndarray) -> np.ndarray:
    T    = len(seq)
    n    = min(MAX_LEN, int(T * 1.25))
    idx  = np.linspace(0, T - 1, n)
    lo   = np.floor(idx).astype(int)
    hi   = np.minimum(lo + 1, T - 1)
    frac = (idx - lo).astype(np.float32)[:, None]
    return (seq[lo] * (1.0 - frac) + seq[hi] * frac).astype(np.float32)

def augment(seq: np.ndarray) -> list:
    variants = [_aug_noise(seq) for _ in range(AUG_COPIES)]
    variants.append(_aug_speedup(seq))
    variants.append(_aug_slowdown(seq))
    return variants


# ── Load sequences ────────────────────────────────────────────
def load_sequences():
    labels_found = sorted([
        d for d in os.listdir(WORD_DATA_DIR)
        if os.path.isdir(os.path.join(WORD_DATA_DIR, d))
    ])
    X, y = [], []
    label_map = {lbl: i for i, lbl in enumerate(labels_found)}

    print("═" * 62)
    print("  WORD DATASET BUILDER")
    print("═" * 62)
    print(f"  {'Label':<20} {'Raw':>6}  {'With aug':>10}")
    print("  " + "─" * 40)

    for label in labels_found:
        folder = os.path.join(WORD_DATA_DIR, label)
        files  = [f for f in os.listdir(folder) if f.endswith(".npy")]
        raw    = 0

        for fname in files:
            seq = np.load(os.path.join(folder, fname))
            if seq.ndim != 2 or seq.shape[0] < MIN_LEN:
                continue

            T, F = seq.shape

            # Backward-compat: old (T,126) data → pad to 302
            if F == 126:
                pad = np.zeros((T, INPUT_SIZE - 126), dtype=np.float32)
                seq = np.concatenate([seq, pad], axis=1)
            elif F != INPUT_SIZE:
                continue

            seq = seq[:MAX_LEN].astype(np.float32)
            X.append(seq); y.append(label_map[label]); raw += 1

            for aug_seq in augment(seq):
                aug_seq = aug_seq[:MAX_LEN]
                if len(aug_seq) >= MIN_LEN:
                    X.append(aug_seq); y.append(label_map[label])

        print(f"  {label:<20} {raw:>6}  {raw * (1 + AUG_COPIES + 2):>10}")

    print("  " + "─" * 40)
    return X, y, labels_found


# ── Pad + mask ────────────────────────────────────────────────
def pad_and_mask(X: list):
    N      = len(X)
    padded = np.zeros((N, MAX_LEN, INPUT_SIZE), dtype=np.float32)
    mask   = np.ones((N, MAX_LEN), dtype=bool)
    for i, seq in enumerate(X):
        T = min(len(seq), MAX_LEN)
        padded[i, :T] = seq[:T]
        mask[i,   :T] = False
    return padded, mask


# ── Main ──────────────────────────────────────────────────────
def build() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_list, y_list, labels_found = load_sequences()

    if not X_list:
        print("\n  ❌ No sequences found. Collect data first.")
        return

    y      = np.array(y_list, dtype=np.int64)
    X_p, masks = pad_and_mask(X_list)

    print(f"\n  Total sequences (incl. aug): {len(X_p)}")
    print(f"  Padded shape              : {X_p.shape}")

    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    np.savez_compressed(
        SAVE_PATH,
        X_train     = X_p[idx_tr],   masks_train = masks[idx_tr],   y_train = y[idx_tr],
        X_test      = X_p[idx_te],   masks_test  = masks[idx_te],   y_test  = y[idx_te],
    )
    with open(LABELS_PATH, "w") as f:
        json.dump(labels_found, f, indent=2)

    print(f"\n  Train: {len(idx_tr)}  |  Test: {len(idx_te)}")
    print(f"  Classes: {len(labels_found)}  →  {labels_found}")
    print(f"\n  Saved → {SAVE_PATH}")
    print(f"  Saved → {LABELS_PATH}")
    print("\n  ✅ Ready for word_train.py")
    print("═" * 62)


if __name__ == "__main__":
    build()
