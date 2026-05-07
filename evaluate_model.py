# ─────────────────────────────────────────────────────────────────────────────
# evaluate_model.py
# Loads the best saved model, evaluates on the test set, prints per-gesture
# accuracy, saves a confusion matrix heatmap, and exports for inference
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import numpy as np
import torch
import torch.nn         as nn
import matplotlib.pyplot as plt
import seaborn          as sns
from sklearn.metrics    import (classification_report,
                                confusion_matrix,
                                accuracy_score)

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'model')
DATASET_PATH = os.path.join(MODEL_DIR, 'dataset_augmented.npz')
LABELS_PATH  = os.path.join(MODEL_DIR, 'labels.json')
MODEL_PATH   = os.path.join(MODEL_DIR, 'asl_model.pt')
CM_PATH      = os.path.join(MODEL_DIR, 'confusion_matrix.png')
EXPORT_PATH  = os.path.join(MODEL_DIR, 'asl_model_inference.pt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── MODEL DEFINITION (must match train_model.py) ──────────────────────────────

class ASLNet(nn.Module):
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
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model():
    checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
    model       = ASLNet(
        input_size  = checkpoint['input_size'],
        num_classes = checkpoint['num_classes'],
        dropout     = checkpoint['dropout'],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f'  ✅  Model loaded  (best epoch: {checkpoint["epoch"]}, '
          f'saved acc: {checkpoint["best_acc"]*100:.2f}%)')
    return model, checkpoint


# ── LOAD TEST DATA ────────────────────────────────────────────────────────────

def load_test_data():
    data  = np.load(DATASET_PATH)
    X     = torch.tensor(data['X_test'], dtype=torch.float32).to(DEVICE)
    y     = data['y_test']

    with open(LABELS_PATH) as f:
        label_map = json.load(f)                          # {"0":"A","1":"B",...}

    class_names = [label_map[str(i)] for i in range(len(label_map))]
    return X, y, class_names


# ── PREDICT ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, X):
    logits = model(X)
    probs  = torch.softmax(logits, dim=1)
    preds  = logits.argmax(dim=1).cpu().numpy()
    confs  = probs.max(dim=1).values.cpu().numpy()
    return preds, confs


# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # normalise rows

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_n,
        annot      = True,
        fmt        = '.2f',
        cmap       = 'Blues',
        xticklabels= class_names,
        yticklabels= class_names,
        linewidths = 0.4,
        ax         = ax,
        vmin       = 0,
        vmax       = 1,
    )
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('True',      fontsize=13)
    ax.set_title('ASL Model — Normalised Confusion Matrix', fontsize=15)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f'  📊  Confusion matrix saved → {CM_PATH}')


# ── PER-GESTURE REPORT ───────────────────────────────────────────────────────

def per_gesture_report(y_true, y_pred, confs, class_names):
    print(f'\n{"═"*58}')
    print('  PER-GESTURE ACCURACY')
    print(f'{"═"*58}')
    print(f'  {"Gesture":<10} {"Correct":>8} {"Total":>7} {"Accuracy":>10} {"Avg Conf":>10}')
    print(f'  {"─"*54}')

    for idx, name in enumerate(class_names):
        mask    = y_true == idx
        correct = int(np.sum(y_pred[mask] == y_true[mask]))
        total   = int(np.sum(mask))
        acc     = correct / total if total > 0 else 0.0
        avg_cf  = float(confs[mask].mean()) if total > 0 else 0.0
        flag    = '  ⚠️' if acc < 0.90 else ''
        print(f'  {name:<10} {correct:>8} {total:>7} {acc*100:>9.1f}% {avg_cf*100:>9.1f}%{flag}')

    print(f'{"═"*58}')


# ── EXPORT FOR INFERENCE ─────────────────────────────────────────────────────

def export_inference_model(model, input_size):
    """
    Export via TorchScript so the inference engine does not need
    the ASLNet class definition at runtime.
    """
    model.eval()
    example  = torch.zeros(1, input_size).to(DEVICE)
    scripted = torch.jit.trace(model, example)
    scripted.save(EXPORT_PATH)
    print(f'  💾  Inference model saved → {EXPORT_PATH}')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f'\n{"═"*58}')
    print('  EVALUATION')
    print(f'{"═"*58}')
    print(f'  Device : {DEVICE}')

    # ── load ──────────────────────────────────────────────
    model, checkpoint = load_model()
    X, y_true, class_names = load_test_data()

    # ── predict ───────────────────────────────────────────
    y_pred, confs = predict(model, X)

    # ── overall accuracy ──────────────────────────────────
    overall = accuracy_score(y_true, y_pred)
    print(f'\n  Overall test accuracy : {overall*100:.2f}%')

    # ── per-gesture breakdown ─────────────────────────────
    per_gesture_report(y_true, y_pred, confs, class_names)

    # ── sklearn classification report ─────────────────────
    print(f'\n{"═"*58}')
    print('  SKLEARN CLASSIFICATION REPORT')
    print(f'{"═"*58}')
    print(classification_report(y_true, y_pred,
                                 target_names=class_names,
                                 digits=3))

    # ── confusion matrix plot ─────────────────────────────
    plot_confusion_matrix(y_true, y_pred, class_names)

    # ── export ────────────────────────────────────────────
    export_inference_model(model, checkpoint['input_size'])

    print(f'\n{"═"*58}')
    print('  FILES SAVED')
    print(f'{"═"*58}')
    print(f'  confusion_matrix.png     ← open to inspect visually')
    print(f'  asl_model_inference.pt   ← used by inference.py')
    print(f'{"═"*58}')
    print('\n✅ Evaluation complete! Ready for Step 8.\n')


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()