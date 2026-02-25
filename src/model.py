"""Transformer backbone for discrete diffusion LM."""

import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep, projected to hidden_dim."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) integer timesteps

        Returns:
            (batch, hidden_dim) timestep embeddings
        """
        half_dim = self.hidden_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch, hidden_dim)
        return self.proj(emb)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with adaptive layer norm for timestep conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        # AdaLN: scale and shift from timestep embedding
        self.adaLN = nn.Linear(hidden_dim, 4 * hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            t_emb: (batch, hidden_dim) timestep embedding

        Returns:
            (batch, seq_len, hidden_dim)
        """
        # Compute adaptive scale/shift for both norms
        ada = self.adaLN(t_emb).unsqueeze(1)  # (batch, 1, 4*hidden_dim)
        scale1, shift1, scale2, shift2 = ada.chunk(4, dim=-1)

        # Self-attention with AdaLN
        h = self.norm1(x) * (1 + scale1) + shift1
        h, _ = self.attn(h, h, h)
        x = x + h

        # FFN with AdaLN
        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.ffn(h)
        x = x + h

        return x


class DiffusionTransformer(nn.Module):
    """Bidirectional transformer for discrete diffusion language modeling.

    Takes corrupted token IDs and a timestep, predicts logits over vocabulary
    for every position.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.timestep_embedding = SinusoidalTimestepEmbedding(hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t: (batch, seq_len) corrupted token IDs
            t: (batch,) integer timesteps

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x_t.shape

        positions = torch.arange(seq_len, device=x_t.device).unsqueeze(0)
        h = self.token_embedding(x_t) + self.position_embedding(positions)
        h = self.drop(h)

        t_emb = self.timestep_embedding(t)  # (batch, hidden_dim)

        for layer in self.layers:
            h = layer(h, t_emb)

        h = self.final_norm(h)
        logits = self.output_head(h)

        return logits
