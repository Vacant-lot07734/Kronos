import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Attention (Full / ProbSparse)
# ---------------------------
class FullAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None):
        # Q,K,V: [B, H, L, D]
        scale = 1.0 / math.sqrt(Q.size(-1))
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B,H,Lq,Lk]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)
        out = torch.matmul(A, V)  # [B,H,Lq,D]
        return out


class ProbAttention(nn.Module):
    """
    简化版 ProbSparse Attention：
    - 对每个 query 采样部分 keys 来估计重要性
    - 选 top-u 的 queries 做精确 attention，其余用 V 的均值近似
    适合长序列；你这里20/7不长，但用户要求Informer，我们按标准实现给出。
    """
    def __init__(self, factor=5, dropout=0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None):
        # Q,K,V: [B, H, L, D]
        B, H, Lq, D = Q.shape
        _, _, Lk, _ = K.shape

        # sample keys
        sample_k = min(Lk, self.factor * int(math.log(Lk + 1)))
        # top queries
        u = min(Lq, self.factor * int(math.log(Lq + 1)))

        # [B,H,Lq,sample_k]
        index_sample = torch.randint(Lk, (Lq, sample_k), device=Q.device)
        K_sample = K[:, :, index_sample, :]  # [B,H,Lq,sample_k,D]
        Q_unsq = Q.unsqueeze(-2)             # [B,H,Lq,1,D]
        # [B,H,Lq,sample_k]
        scores_sample = torch.matmul(Q_unsq, K_sample.transpose(-2, -1)).squeeze(-2)

        # sparsity measurement
        M = scores_sample.max(dim=-1).values - scores_sample.mean(dim=-1)  # [B,H,Lq]
        top_u = M.topk(u, dim=-1).indices  # [B,H,u]

        # build context: mean(V) for all queries
        V_mean = V.mean(dim=-2, keepdim=True)  # [B,H,1,D]
        context = V_mean.repeat(1, 1, Lq, 1).contiguous()  # [B,H,Lq,D]

        # exact attention for top_u queries
        # gather Q_top: [B,H,u,D]
        Q_top = torch.gather(Q, dim=2, index=top_u.unsqueeze(-1).expand(-1, -1, -1, D))
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(Q_top, K.transpose(-2, -1)) * scale  # [B,H,u,Lk]
        if attn_mask is not None:
            # attn_mask: [B,1,Lq,Lk] or [B,H,Lq,Lk]
            # 选出 top_u 对应的 mask 行：mask_top -> [B,maskH,u,Lk]
            if attn_mask.dim() != 4:
                raise ValueError(f"attn_mask must be 4D, got {attn_mask.shape}")

            if attn_mask.size(1) == 1:
                # [B,1,Lq,Lk] -> [B,1,u,Lk]
                mask_top = attn_mask[:, :, top_u, :]
            else:
                # [B,H,Lq,Lk] -> [B,H,u,Lk]
                mask_top = attn_mask[:, :, top_u, :]

            scores = scores.masked_fill(mask_top, float("-inf"))

        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)
        out_top = torch.matmul(A, V)  # [B,H,u,D]

        # scatter back
        context.scatter_(2, top_u.unsqueeze(-1).expand(-1, -1, -1, D), out_top)
        return context


class AttentionLayer(nn.Module):
    def __init__(self, attn, d_model, n_heads):
        super().__init__()
        self.attn = attn
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def _split(self, x):
        # x: [B,L,d_model] -> [B,H,L,d_head]
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, self.d_head).transpose(1, 2).contiguous()
        return x

    def _merge(self, x):
        # x: [B,H,L,d_head] -> [B,L,d_model]
        B, H, L, Dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return x

    def forward(self, q, k, v, attn_mask=None):
        Q = self._split(self.Wq(q))
        K = self._split(self.Wk(k))
        V = self._split(self.Wv(v))
        out = self.attn(Q, K, V, attn_mask=attn_mask)
        out = self._merge(out)
        return self.out(out)


# ---------------------------
# Encoder / Decoder blocks
# ---------------------------
class EncoderLayer(nn.Module):
    def __init__(self, attn_layer, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.attn = attn_layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: [B,L,d_model]
        a = self.attn(x, x, x, attn_mask)
        x = self.norm1(x + self.drop(a))
        f = self.ff(x)
        x = self.norm2(x + self.drop(f))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attn_layer, cross_attn_layer, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = self_attn_layer
        self.cross_attn = cross_attn_layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        a1 = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.drop(a1))
        a2 = self.cross_attn(x, enc_out, enc_out, cross_mask)
        x = self.norm2(x + self.drop(a2))
        f = self.ff(x)
        x = self.norm3(x + self.drop(f))
        return x


def causal_mask(B, L, device):
    # [B,1,L,L]
    mask = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)


# ---------------------------
# Informer Model
# ---------------------------
class Informer(nn.Module):
    """
    输入:
      x_enc: [B, lookback, 6]
      x_dec: [B, label_len + pred_len, 6] (前label_len是历史，后pred_len是0占位)
    输出:
      y_hat: [B, pred_len, 6]
    """
    def __init__(
        self,
        input_dim=6,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        dropout=0.1,
        attn="prob",
        factor=5,
        pred_len=7,
        label_len=10,
        output_dim=6
    ):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.output_dim = output_dim

        # value embedding (no extra time embedding version)
        self.enc_in = nn.Linear(input_dim, d_model)
        self.dec_in = nn.Linear(input_dim, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout)

        Attn = ProbAttention(factor=factor, dropout=dropout) if attn == "prob" else FullAttention(dropout=dropout)

        # Encoder
        enc_layers = []
        for _ in range(e_layers):
            attn_layer = AttentionLayer(Attn, d_model, n_heads)
            enc_layers.append(EncoderLayer(attn_layer, d_model, d_ff, dropout))
        self.encoder = nn.ModuleList(enc_layers)

        # Decoder (self-attn + cross-attn)
        dec_layers = []
        for _ in range(d_layers):
            self_attn_layer = AttentionLayer(Attn, d_model, n_heads)
            cross_attn_layer = AttentionLayer(FullAttention(dropout=dropout), d_model, n_heads)
            dec_layers.append(DecoderLayer(self_attn_layer, cross_attn_layer, d_model, d_ff, dropout))
        self.decoder = nn.ModuleList(dec_layers)

        self.proj = nn.Linear(d_model, output_dim)

    def forward(self, x_enc, x_dec):
        # x_enc: [B, L_enc, 6]
        # x_dec: [B, L_dec, 6]
        B = x_enc.size(0)

        enc = self.pos_emb(self.enc_in(x_enc))
        for layer in self.encoder:
            enc = layer(enc, attn_mask=None)

        dec = self.pos_emb(self.dec_in(x_dec))
        self_mask = causal_mask(B, dec.size(1), dec.device)  # decoder causal
        for layer in self.decoder:
            dec = layer(dec, enc_out=enc, self_mask=self_mask, cross_mask=None)

        out = self.proj(dec)  # [B, L_dec, 6]
        return out[:, -self.pred_len:, :]  # 取最后pred_len


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,max_len,d_model]

    def forward(self, x):
        # x: [B,L,D]
        x = x + self.pe[:, : x.size(1), :]
        return self.drop(x)
