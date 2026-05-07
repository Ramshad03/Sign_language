# ─────────────────────────────────────────────────────────────────────────────
# build_dataset.py
# Loads all collected .npy landmark files, encodes labels, splits into
# train/test sets, and saves a single dataset.npz file for training
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
import json

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR     = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR      = os.path.join(os.path.dirname(__file__), 'model')
DATASET_PATH = os.path.join(OUT_DIR, 'dataset.npz')
LABELS_PATH  = os.path.join(OUT_DIR, 'labels.json')
TEST_SIZE    = 0.15       # 15% held out for testing
RANDOM_SEED  = 42

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_dataset():
    """
    Walk every gesture folder inside DATA_DIR.
    Each .npy file → one row in X (63 floats).
    Folder name    → one entry in y (string label).
    """
    X, y = [], []

    gesture_dirs = sorted(os.listdir(DATA_DIR))

    print(f'\n{"═"*52}')
    print('  LOADING DATASET')
    print(f'{"═"*52}')

    for gesture in gesture_dirs:
        gesture_path = os.path.join(DATA_DIR, gesture)

        if not os.path.isdir(gesture_path):
            continue

        files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]

        if len(files) == 0:
            print(f'  ⚠️  {gesture:8s} — no samples found, skipping')
            continue

        for fname in files:
            fpath     = os.path.join(gesture_path, fname)
            landmarks = np.load(fpath)

            # ── sanity check: every sample must be 63 values ──
            if landmarks.shape != (63,):
                print(f'  ⚠️  Bad shape {landmarks.shape} in {fpath}, skipping')
                continue

            X.append(landmarks)
            y.append(gesture)

        print(f'  ✅  {gesture:8s} — {len(files):4d} samples loaded')

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y


# ── ENCODE + SPLIT ────────────────────────────────────────────────────────────

def encode_and_split(X, y):
    """
    Label-encode string gestures → integers.
    Split into stratified train / test sets.
    """
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)          # e.g. 'A'→0, 'B'→1 …

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = y_enc              # keeps class balance in both splits
    )

    return X_train, X_test, y_train, y_test, le


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save(X_train, X_test, y_train, y_test, le):
    """
    Save arrays as a single compressed .npz file.
    Save label mapping as labels.json for use during inference.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── compressed numpy archive ──────────────────────────
    np.savez_compressed(
        DATASET_PATH,
        X_train = X_train,
        X_test  = X_test,
        y_train = y_train,
        y_test  = y_test,
    )

    # ── label map: index → gesture string ─────────────────
    label_map = {int(i): str(label) for i, label in enumerate(le.classes_)}
    with open(LABELS_PATH, 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f'\n  💾  Dataset  saved → {DATASET_PATH}')
    print(f'  💾  Label map saved → {LABELS_PATH}')


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def summary(X_train, X_test, y_train, y_test, le):
    print(f'\n{"═"*52}')
    print('  DATASET SUMMARY')
    print(f'{"═"*52}')
    print(f'  Total gestures  : {len(le.classes_)}')
    print(f'  Feature size    : {X_train.shape[1]} (21 landmarks × 3)')
    print(f'  Training samples: {len(X_train)}')
    print(f'  Test samples    : {len(X_test)}')
    print(f'  Label map       :')
    for idx, label in enumerate(le.classes_):
        print(f'      {idx:2d} → {label}')
    print(f'{"═"*52}')


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # 1 — load all .npy files
    X, y = load_dataset()

    if len(X) == 0:
        print('\n❌ No data found. Run collect_data.py first.')
        exit(1)

    # 2 — encode labels + split
    X_train, X_test, y_train, y_test, le = encode_and_split(X, y)

    # 3 — save to disk
    save(X_train, X_test, y_train, y_test, le)

    # 4 — print summary
    summary(X_train, X_test, y_train, y_test, le)

    print('\n✅ Dataset built successfully! Ready for Step 5.\n')
    