# ══════════════════════════════════════════════════════════════
# word_train.py — Train GestureTransformer on variable-length sequences
# ══════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import os
import matplotlib.pyplot as plt

from gesture_model import GestureTransformer, save_checkpoint, load_checkpoint

# ── Config ────────────────────────────────────────────────────
DATASET_PATH  = os.path.join("model", "word_dataset.npz")
LABELS_PATH   = os.path.join("model", "word_labels.json")
MODEL_DIR     = "model"
SAVE_PATH     = os.path.join(MODEL_DIR, "word_model.pt")
PLOT_PATH     = os.path.join(MODEL_DIR, "lstm_training_curve.png")

MAX_LEN       = 90
INPUT_SIZE    = 302
D_MODEL       = 128
N_HEAD        = 4
N_LAYERS      = 3
DROPOUT_TRAIN = 0.20
EPOCHS        = 120
BATCH_SIZE    = 32
LR            = 5e-4
WEIGHT_DECAY  = 1e-4


# ── Data loading ──────────────────────────────────────────────
def load_data(device):
    data   = np.load(DATASET_PATH)
    labels = json.load(open(LABELS_PATH))

    def tensors(split):
        X    = torch.tensor(data[f"X_{split}"],     dtype=torch.float32)
        mask = torch.tensor(data[f"masks_{split}"], dtype=torch.bool)
        y    = torch.tensor(data[f"y_{split}"],     dtype=torch.long)
        return TensorDataset(X, mask, y)

    train_ds = tensors("train")
    test_ds  = tensors("test")
    pin      = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          num_workers=0, pin_memory=pin)
    return train_dl, test_dl, labels


# ── Evaluation ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for X, mask, y in loader:
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        correct += (model(X, mask).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total * 100


@torch.no_grad()
def per_class_accuracy(model, loader, labels, device):
    model.eval()
    n = len(labels)
    correct = np.zeros(n, dtype=int)
    total   = np.zeros(n, dtype=int)
    for X, mask, y in loader:
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        preds = model(X, mask).argmax(1).cpu()
        for pred, gt in zip(preds, y.cpu()):
            total[gt]   += 1
            correct[gt] += int(pred == gt)
    print("\n  Per-class accuracy:")
    print(f"  {'Class':<20} {'Acc':>8}  {'N':>6}")
    print("  " + "─" * 38)
    for i, lbl in enumerate(labels):
        acc = correct[i] / total[i] * 100 if total[i] > 0 else 0.0
        print(f"  {lbl:<20} {acc:>7.1f}%  {total[i]:>6}")


# ── Training ──────────────────────────────────────────────────
def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("═" * 62)
    print("  GESTURE TRANSFORMER TRAINING")
    print("═" * 62)
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    train_dl, test_dl, labels = load_data(device)
    num_classes = len(labels)
    print(f"  Classes    : {num_classes}")
    print(f"  Train seqs : {len(train_dl.dataset)}")
    print(f"  Test  seqs : {len(test_dl.dataset)}")
    print("═" * 62)

    model_cfg = dict(
        input_size  = INPUT_SIZE,
        num_classes = num_classes,
        d_model     = D_MODEL,
        nhead       = N_HEAD,
        num_layers  = N_LAYERS,
        dropout     = 0.0,        # stored as 0 for clean inference loading
        max_len     = MAX_LEN,
    )
    model     = GestureTransformer(**{**model_cfg, "dropout": DROPOUT_TRAIN}).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,}")
    print("═" * 62)

    history    = {"train_acc": [], "test_acc": []}
    best_acc   = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, mask, y in train_dl:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X, mask), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            tr_acc = evaluate(model, train_dl, device)
            te_acc = evaluate(model, test_dl,  device)
            history["train_acc"].append(tr_acc)
            history["test_acc"].append(te_acc)

            flag = ""
            if te_acc > best_acc:
                best_acc   = te_acc
                best_epoch = epoch
                save_checkpoint(model, labels, best_acc, SAVE_PATH, model_cfg)
                flag = "  ← best"

            print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                  f"train {tr_acc:5.1f}%  test {te_acc:5.1f}%  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}{flag}")

    # Training curve
    epochs_logged = [1] + list(range(10, EPOCHS + 1, 10))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs_logged[:len(history["train_acc"])], history["train_acc"], label="Train")
    ax.plot(epochs_logged[:len(history["test_acc"])],  history["test_acc"],  label="Test")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("GestureTransformer")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOT_PATH, dpi=150); plt.close()

    # Per-class breakdown
    best_model, _ = load_checkpoint(SAVE_PATH, device)
    per_class_accuracy(best_model, test_dl, labels, device)

    print(f"\n{'═' * 62}")
    print(f"  Best test accuracy : {best_acc:.1f}%  (epoch {best_epoch})")
    print(f"  Saved → {SAVE_PATH}")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    train()
