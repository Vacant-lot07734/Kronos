"""
Lightweight Transformer encoder for hourly bar sequences.

Encodes hourly OHLCVA bars into continuous hidden states that serve as
conditional context for the daily predictor via cross-attention fusion.
"""

import sys
import torch
import torch.nn as nn

sys.path.append("../")
from model.module import TransformerBlock, TemporalEmbedding, RMSNorm


class HourlyEncoder(nn.Module):
    """
    Encodes hourly bar data into a sequence of hidden states.

    Input:
        x_hourly   (B, L_h, d_in)   -- normalised hourly OHLCVA features
        x_stamp_h  (B, L_h, 5)      -- time features [minute, hour, weekday, day, month]

    Output:
        hourly_hidden  (B, L_h, d_model)
    """

    def __init__(
        self,
        d_in: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        ff_dim: int = 1024,
        n_layers: int = 2,
        ffn_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        learn_te: bool = False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model

        self.input_proj = nn.Linear(d_in, d_model)
        self.time_emb = TemporalEmbedding(d_model, learn_te)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

    def forward(self, x_hourly: torch.Tensor, x_stamp_h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_hourly:  (B, L_h, d_in)
            x_stamp_h: (B, L_h, 5)
        Returns:
            (B, L_h, d_model)
        """
        h = self.input_proj(x_hourly)
        h = h + self.time_emb(x_stamp_h)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return h
