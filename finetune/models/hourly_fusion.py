"""
Cross-attention fusion layer that injects hourly context into daily
predictor hidden states.

    H_d' = H_d + CrossAttn(Q=H_d, K=H_h, V=H_h)
"""

import sys
import torch
import torch.nn as nn

sys.path.append("../")
from model.module import RMSNorm, MultiHeadCrossAttentionWithRoPE


class HourlyFusionLayer(nn.Module):
    """
    Pre-norm cross-attention fusion with residual connection.

    Input:
        daily_hidden   (B, T_d, d_model) -- daily predictor hidden states
        hourly_hidden  (B, L_h, d_model) -- hourly encoder output

    Output:
        fused  (B, T_d, d_model)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
    ):
        super().__init__()
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(
            d_model, n_heads, attn_dropout_p, resid_dropout_p
        )

    def forward(
        self,
        daily_hidden: torch.Tensor,
        hourly_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            daily_hidden:  (B, T_d, d_model)
            hourly_hidden: (B, L_h, d_model)
        Returns:
            (B, T_d, d_model)
        """
        q = self.norm_q(daily_hidden)
        kv = self.norm_kv(hourly_hidden)
        attn_out = self.cross_attn(query=q, key=kv, value=kv)
        return daily_hidden + attn_out
