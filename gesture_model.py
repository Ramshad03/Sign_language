# ══════════════════════════════════════════════════════════════
# gesture_model.py — GestureTransformer
#
# Replaces the fixed-length Bidirectional LSTM with a Transformer
# encoder that handles variable-length gesture segments naturally
# via padding masks and masked mean-pooling.
#
# Why Transformer over LSTM for this task:
#   • Attention lets the model relate frame 1 to frame 25 directly —
#     no information decay through recurrent state.
#   • Masking makes variable-length sequences first-class; the model
#     learns gestures independent of signing speed.
#   • Pre-LN (norm_first=True) makes training more stable without
#     careful LR tuning.
#   • Mean-pooling over unmasked timesteps beats last-timestep and
#     correctly exploits the full bidirectional context.
#
# Input  : (B, T, input_size)  — padded feature sequences
# Mask   : (B, T)  bool        — True where position is padding
# Output : (B, num_classes)    — raw logits
# ══════════════════════════════════════════════════════════════

from __future__ import annotations
import torch
import torch.nn as nn
import os
import json
from typing import Optional


class GestureTransformer(nn.Module):
    """
    Lightweight Transformer encoder for variable-length gesture classification.

    Args:
        input_size:  Feature dimension per frame (302 for two-hand rich features).
        num_classes: Number of gesture/word classes.
        d_model:     Internal embedding dimension.
        nhead:       Number of attention heads (must divide d_model evenly).
        num_layers:  Number of Transformer encoder layers.
        dropout:     Dropout rate inside attention and FFN.
        max_len:     Maximum sequence length (used for positional embedding).
    """

    def __init__(
        self,
        input_size:  int,
        num_classes: int,
        d_model:     int = 128,
        nhead:       int = 4,
        num_layers:  int = 3,
        dropout:     float = 0.2,
        max_len:     int = 90,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embed  = nn.Embedding(max_len, d_model)
        self._max_len   = max_len

        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= d_model * 4,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers,
                                                  enable_nested_tensor=False)
        self.norm        = nn.LayerNorm(d_model)
        self.head        = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, input_size)
            mask: (B, T) bool — True marks padded (ignored) positions.
                  Pass None when all positions are valid.
        Returns:
            logits (B, num_classes)
        """
        B, T, _ = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x    = self.input_proj(x) + self.pos_embed(pos)
        x    = self.transformer(x, src_key_padding_mask=mask)

        # Masked mean-pooling: average only over real (non-padded) positions
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)       # (B, T, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        return self.head(self.norm(x))


# ── Save / load helpers ───────────────────────────────────────
def save_checkpoint(
    model:      GestureTransformer,
    labels:     list,
    best_acc:   float,
    path:       str,
    config:     dict,
) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "config":      config,
        "labels":      labels,
        "best_acc":    best_acc,
    }, path)


def load_checkpoint(path: str, device: torch.device) -> tuple:
    """
    Load a saved GestureTransformer checkpoint.

    Returns:
        model   — GestureTransformer in eval mode
        labels  — list of class name strings
    """
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    cfg    = ckpt["config"]
    model  = GestureTransformer(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model, ckpt["labels"]
