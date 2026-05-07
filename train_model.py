# ─────────────────────────────────────────────────────────────────────────────
# train_model.py
# Builds, trains and saves the ASL gesture MLP classifier using PyTorch
# Architecture: 63 → 512 → 256 → 128 → 29 (gestures)
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import numpy as np
import torch
import torch.nn            as nn
import torch.optim         as optim
from torch.utils.data      import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot   as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'model')
DATASET_PATH = os.path.join(MODEL_DIR, 'dataset_augmented.npz')
LABELS_PATH  = os.path.join(MODEL_DIR, 'labels.json')
MODEL_PATH   = os.path.join(MODEL_DIR, 'asl_model.pt')
PLOT_PATH    = os.path.join(MODEL_DIR, 'training_curve.png')

BATCH_SIZE   = 128
EPOCHS       = 80
LR           = 1e-3        # initial learning rate
DROPOUT      = 0.4
RANDOM_SEED  = 42

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── DEVICE ────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n  Device : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  GPU    : {torch.cuda.get_device_name(0)}')

# ── MODEL ─────────────────────────────────────────────────────────────────────

class ASLNet(nn.Module):
    """
    Four-layer MLP for static ASL gesture classification.

    63 input features  (21 landmarks × 3 normalised coords)
    512 hidden units   layer 1
    256 hidden units   layer 2
    128 hidden units   layer 3
    N   output classes (one per gesture)

    Each hidden layer: Linear → BatchNorm → ReLU → Dropout
    """

    def __init__(self, input_size: int, num_classes: int, dropout: float):
        super().__init__()

        def block(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.net = nn.Sequential(
            block(input_size, 512),
            block(512,        256),
            block(256,        128),
            nn.Linear(128, num_classes),   # final layer — no activation (CrossEntropy handles it)
        )

    def forward(self, x):
        return self.net(x)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data():
    data    = np.load(DATASET_PATH)
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    X_test  = torch.tensor(data['X_test'],  dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.long)
    y_test  = torch.tensor(data['y_test'],  dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(json.load(open(LABELS_PATH)))
    input_size  = X_train.shape[1]

    return train_loader, test_loader, input_size, num_classes


# ── TRAIN ONE EPOCH ───────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimiser):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimiser.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ── EVALUATE ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        logits      = model(X_batch)
        loss        = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ── PLOT TRAINING CURVES ──────────────────────────────────────────────────────

def plot_curves(history):
    epochs = range(1, len(history['train_acc']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── accuracy ──────────────────────────────────────────
    ax1.plot(epochs, history['train_acc'], label='Train')
    ax1.plot(epochs, history['val_acc'],   label='Test')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── loss ──────────────────────────────────────────────
    ax2.plot(epochs, history['train_loss'], label='Train')
    ax2.plot(epochs, history['val_loss'],   label='Test')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f'  📊  Training curve saved → {PLOT_PATH}')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f'\n{"═"*52}')
    print('  LOADING DATA')
    print(f'{"═"*52}')

    train_loader, test_loader, input_size, num_classes = load_data()

    print(f'  Input size   : {input_size}')
    print(f'  Num classes  : {num_classes}')
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Test batches : {len(test_loader)}')

    # ── build model ───────────────────────────────────────
    model     = ASLNet(input_size, num_classes, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, mode='max', patience=6,
                                  factor=0.5, min_lr=1e-5)

    print(f'\n{"═"*52}')
    print('  MODEL ARCHITECTURE')
    print(f'{"═"*52}')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'\n  Trainable parameters: {total_params:,}')

    # ── training loop ─────────────────────────────────────
    print(f'\n{"═"*52}')
    print('  TRAINING')
    print(f'{"═"*52}')

    history     = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc    = 0.0
    best_epoch  = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimiser)
        va_loss, va_acc = evaluate(model, test_loader,  criterion)

        scheduler.step(va_acc)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        # ── save best model ───────────────────────────────
        if va_acc > best_acc:
            best_acc   = va_acc
            best_epoch = epoch
            torch.save({
                'epoch'       : epoch,
                'model_state' : model.state_dict(),
                'input_size'  : input_size,
                'num_classes' : num_classes,
                'dropout'     : DROPOUT,
                'best_acc'    : best_acc,
            }, MODEL_PATH)

        # ── print every 5 epochs ──────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]['lr']
            flag   = '  ← best' if epoch == best_epoch else ''
            print(f'  Epoch {epoch:3d}/{EPOCHS}  '
                  f'train acc: {tr_acc:.4f}  '
                  f'test acc: {va_acc:.4f}  '
                  f'lr: {lr_now:.6f}{flag}')

    # ── final summary ─────────────────────────────────────
    plot_curves(history)

    print(f'\n{"═"*52}')
    print('  TRAINING COMPLETE')
    print(f'{"═"*52}')
    print(f'  Best test accuracy : {best_acc*100:.2f}%  (epoch {best_epoch})')
    print(f'  Model saved        → {MODEL_PATH}')
    print(f'{"═"*52}')
    print('\n✅ Training done! Ready for Step 7.\n')


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()