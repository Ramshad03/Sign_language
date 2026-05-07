# ─────────────────────────────────────────────────────────────────────────────
# preprocess.py
# Normalises landmarks relative to the wrist, augments the training set,
# and saves dataset_augmented.npz ready for model training
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_DIR     = os.path.join(os.path.dirname(__file__), 'model')
DATASET_IN    = os.path.join(MODEL_DIR, 'dataset.npz')
DATASET_OUT   = os.path.join(MODEL_DIR, 'dataset_augmented.npz')

AUGMENT_TIMES = 5       # generate N extra copies per training sample
NOISE_STD     = 0.005   # gaussian noise strength (keep small — 0.003–0.008)
RANDOM_SEED   = 42

rng = np.random.default_rng(RANDOM_SEED)

# ── NORMALISATION ─────────────────────────────────────────────────────────────

def normalise(X):
    """
    For each sample (63 values = 21 landmarks × 3):
      1. Re-centre: subtract wrist (landmark 0) x,y,z from every landmark
      2. Scale: divide by the max absolute value so range fits in -1 to 1

    This makes the model invariant to hand position and distance from camera.
    """
    X = X.copy()
    out = np.zeros_like(X)

    for i, sample in enumerate(X):
        # reshape to (21, 3) for easy manipulation
        lm = sample.reshape(21, 3)

        # step 1 — re-centre around wrist (landmark index 0)
        lm = lm - lm[0]

        # step 2 — scale to [-1, 1]
        max_val = np.max(np.abs(lm))
        if max_val > 0:
            lm = lm / max_val

        out[i] = lm.flatten()

    return out


# ── AUGMENTATION ──────────────────────────────────────────────────────────────

def augment(X_train, y_train):
    """
    For each training sample, create AUGMENT_TIMES extra copies by:
      - Adding small gaussian noise to x, y, z values
      - Mirroring x-axis (simulates left hand / different angle)
    Returns the original samples PLUS all augmented ones concatenated.
    """
    aug_X, aug_y = [X_train], [y_train]

    for _ in range(AUGMENT_TIMES):

        # ── gaussian noise ────────────────────────────────
        noise   = rng.normal(0, NOISE_STD, X_train.shape).astype(np.float32)
        noisy   = np.clip(X_train + noise, -1.0, 1.0)
        aug_X.append(noisy)
        aug_y.append(y_train)

        # ── mirror x-axis ─────────────────────────────────
        # every 3rd value starting at index 0 is the x coordinate
        mirrored = X_train.copy()
        mirrored[:, 0::3] = -mirrored[:, 0::3]
        aug_X.append(mirrored)
        aug_y.append(y_train)

    X_aug = np.concatenate(aug_X, axis=0)
    y_aug = np.concatenate(aug_y, axis=0)

    # ── shuffle ───────────────────────────────────────────
    indices = rng.permutation(len(X_aug))
    return X_aug[indices], y_aug[indices]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f'\n{"═"*52}')
    print('  PREPROCESSING & AUGMENTATION')
    print(f'{"═"*52}')

    # ── load ──────────────────────────────────────────────
    data    = np.load(DATASET_IN)
    X_train = data['X_train'].astype(np.float32)
    X_test  = data['X_test'].astype(np.float32)
    y_train = data['y_train']
    y_test  = data['y_test']

    print(f'\n  Loaded training samples : {len(X_train)}')
    print(f'  Loaded test samples     : {len(X_test)}')

    # ── normalise ─────────────────────────────────────────
    print('\n  Normalising landmarks...')
    X_train = normalise(X_train)
    X_test  = normalise(X_test)
    print('  ✅ Normalisation done')

    # ── augment training set only ─────────────────────────
    # NOTE: we never augment the test set — it must stay real
    print(f'\n  Augmenting training set ({AUGMENT_TIMES}× noise + mirror)...')
    X_train_aug, y_train_aug = augment(X_train, y_train)
    print(f'  ✅ Augmentation done')
    print(f'     Before : {len(X_train):6d} samples')
    print(f'     After  : {len(X_train_aug):6d} samples')

    # ── save ──────────────────────────────────────────────
    np.savez_compressed(
        DATASET_OUT,
        X_train = X_train_aug,
        X_test  = X_test,
        y_train = y_train_aug,
        y_test  = y_test,
    )

    print(f'\n  💾  Saved → {DATASET_OUT}')

    # ── final summary ─────────────────────────────────────
    print(f'\n{"═"*52}')
    print('  FINAL DATASET SUMMARY')
    print(f'{"═"*52}')
    print(f'  Feature shape    : ({X_train_aug.shape[1]},) — 21 × 3 normalised')
    print(f'  Training samples : {len(X_train_aug)}')
    print(f'  Test samples     : {len(X_test)}')
    print(f'  Value range      : [{X_train_aug.min():.3f}, {X_train_aug.max():.3f}]')
    print(f'{"═"*52}')
    print('\n✅ Preprocessing complete! Ready for Step 6.\n')


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()