"""
Transformer Encoder Sequence Classifier
========================================
Binary classification of individual elements in a sequence of vectors.
Each element is classified using the full context of the sequence via
multi-head self-attention in a standard transformer encoder stack.

Input:  (batch_size, seq_len, input_dim)  — a batch of sequences of vectors
Output: (batch_size, seq_len)             — per-element binary logits
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)            # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                          # (d_model/2,)

        pe = torch.zeros(1, max_len, d_model)                    # (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerSequenceClassifier(nn.Module):
    """
    Per-element binary classifier that uses a transformer encoder to incorporate
    full sequence context before making each element's classification decision.

    Architecture:
        1. Linear projection:  input_dim → d_model
        2. Positional encoding (sinusoidal)
        3. TransformerEncoder  (num_encoder_layers × TransformerEncoderLayer)
        4. Classifier head:    d_model → 1  (raw logit per element)

    Loss: use nn.BCEWithLogitsLoss on the raw logits.
    """

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim:          Dimensionality of each input vector.
            d_model:            Internal model dimension (must be divisible by nhead).
            nhead:              Number of self-attention heads.
            num_encoder_layers: Number of stacked TransformerEncoderLayer blocks.
            dim_feedforward:    Hidden size of the per-layer feed-forward network.
            dropout:            Dropout probability used throughout the model.
        """
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # expects (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) — input sequence of vectors
            src_key_padding_mask: (batch_size, seq_len) bool tensor where True
                marks positions that should be ignored (padding). Optional.

        Returns:
            logits: (batch_size, seq_len) — raw binary logit for each element.
                    Pass to nn.BCEWithLogitsLoss for training, or apply
                    torch.sigmoid for probabilities.
        """
        x = self.input_projection(x)          # (B, L, d_model)
        x = self.positional_encoding(x)       # (B, L, d_model)
        x = self.transformer_encoder(         # (B, L, d_model)
            x, src_key_padding_mask=src_key_padding_mask
        )
        logits = self.classifier(x)           # (B, L, 1)
        return logits.squeeze(-1)             # (B, L)


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Hyperparameters
    BATCH_SIZE = 8
    SEQ_LEN = 20
    INPUT_DIM = 16
    NUM_STEPS = 100
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = TransformerSequenceClassifier(
        input_dim=INPUT_DIM,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Verify output shape
    x_demo = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, device=device)
    with torch.no_grad():
        out = model(x_demo)
    assert out.shape == (BATCH_SIZE, SEQ_LEN), f"Unexpected shape: {out.shape}"
    print(f"Forward pass OK — output shape: {out.shape}")

    # Short training loop on random data
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print("\nTraining on random data:")
    print(f"{'Step':>6}  {'Loss':>8}  {'Accuracy':>9}")
    print("-" * 30)

    for step in range(1, NUM_STEPS + 1):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, device=device)
        labels = torch.randint(0, 2, (BATCH_SIZE, SEQ_LEN), dtype=torch.float, device=device)

        logits = model(x)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == labels).float().mean().item()
            print(f"{step:>6}  {loss.item():>8.4f}  {acc:>8.1%}")

    print("\nDone.")
