"""
Cross-attention fusion layer that injects hourly context into daily
predictor hidden states.

    H_d' = H_d + CrossAttn(Q=H_d, K=H_h, V=H_h)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")
from model.module import RMSNorm


class MultiHeadCrossAttention(nn.Module):
    """Plain cross-attention for heterogeneous daily/hourly sequence lengths.

    C-group fusion queries daily hidden states with hourly hidden states as
    memory. The two streams naturally have different sequence lengths, so the
    fusion layer should not reuse the predictor's RoPE-based attention, whose
    current implementation assumes ``q_len == k_len``.
    """

    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, q_len, -1)
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


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
        self.cross_attn = MultiHeadCrossAttention(
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
