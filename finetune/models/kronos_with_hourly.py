"""
Composite model for C-group: Kronos daily predictor + hourly encoder + fusion.

Architecture:
    1. tokenizer.encode(x_daily)  →  (s1_ids, s2_ids)
    2. kronos.embedding + time_emb  →  x
    3. kronos.transformer[:split]   →  lower_hidden   (frozen)
    4. hourly_encoder(x_hourly)     →  hourly_hidden
    5. fusion(lower_hidden, hourly_hidden) → fused
    6. kronos.transformer[split:]   →  upper_hidden   (trainable)
    7. kronos.norm → head           →  logits
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")
from model.kronos import Kronos
from finetune.models.hourly_encoder import HourlyEncoder
from finetune.models.hourly_fusion import HourlyFusionLayer


class KronosWithHourly(nn.Module):
    """
    End-to-end C-group model.  Wraps the pretrained Kronos predictor and
    adds hourly encoder + cross-attention fusion.

    The Kronos transformer layers are split at ``split_point``:
      - layers [0, split_point)  are frozen  (lower)
      - layers [split_point, n)  are trainable (upper)

    Fusion is inserted between the two halves.
    """

    def __init__(
        self,
        kronos: Kronos,
        hourly_encoder: HourlyEncoder,
        fusion_layer: HourlyFusionLayer,
        split_point: int,
    ):
        super().__init__()
        self.kronos = kronos
        self.hourly_encoder = hourly_encoder
        self.fusion_layer = fusion_layer
        self.split_point = split_point
        self.n_layers = len(kronos.transformer)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, s1_ids, s2_ids, stamp_d, x_hourly, x_stamp_h,
                padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        """
        Full forward pass used during training (teacher forcing).

        Args:
            s1_ids, s2_ids: (B, T)  daily token ids
            stamp_d:        (B, T, 5)  daily time features
            x_hourly:       (B, L_h, d_in)  normalised hourly features
            x_stamp_h:      (B, L_h, 5)     hourly time features
        Returns:
            (s1_logits, s2_logits)
        """
        # --- hourly branch (always requires grad from encoder/fusion) ---
        hourly_hidden = self.hourly_encoder(x_hourly, x_stamp_h)

        # --- daily branch ---
        x = self.kronos.embedding([s1_ids, s2_ids])
        if stamp_d is not None:
            x = x + self.kronos.time_emb(stamp_d)
        x = self.kronos.token_drop(x)

        # lower transformer layers (frozen)
        for layer in self.kronos.transformer[:self.split_point]:
            x = layer(x, key_padding_mask=padding_mask)

        # cross-attention fusion
        x = self.fusion_layer(x, hourly_hidden)

        # upper transformer layers (trainable)
        for layer in self.kronos.transformer[self.split_point:]:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.kronos.norm(x)

        # dual-head decoding
        s1_logits = self.kronos.head(x)

        if use_teacher_forcing:
            sibling_embed = self.kronos.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1 = torch.multinomial(
                s1_probs.view(-1, self.kronos.s1_vocab_size), 1
            ).view(s1_ids.shape)
            sibling_embed = self.kronos.embedding.emb_s1(sample_s1)

        x2 = self.kronos.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.kronos.head.cond_forward(x2)

        return s1_logits, s2_logits

    # ------------------------------------------------------------------
    # Auto-regressive inference helpers
    # ------------------------------------------------------------------

    def encode_hourly(self, x_hourly: torch.Tensor, x_stamp_h: torch.Tensor) -> torch.Tensor:
        """Encode hourly context once; reuse across AR steps."""
        return self.hourly_encoder(x_hourly, x_stamp_h)

    def decode_s1(self, s1_ids, s2_ids, stamp_d, hourly_hidden, padding_mask=None):
        """
        Predict s1 logits + return context for a single AR step.
        ``hourly_hidden`` should be pre-computed via ``encode_hourly``.
        """
        x = self.kronos.embedding([s1_ids, s2_ids])
        if stamp_d is not None:
            x = x + self.kronos.time_emb(stamp_d)
        x = self.kronos.token_drop(x)

        for layer in self.kronos.transformer[:self.split_point]:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.fusion_layer(x, hourly_hidden)

        for layer in self.kronos.transformer[self.split_point:]:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.kronos.norm(x)
        s1_logits = self.kronos.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        """Decode s2 tokens conditioned on context and s1 ids."""
        return self.kronos.decode_s2(context, s1_ids, padding_mask)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_c_modules(self, save_dir: str):
        """Save only the C-group specific modules (encoder + fusion)."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.hourly_encoder.state_dict(), os.path.join(save_dir, "hourly_encoder.pt"))
        torch.save(self.fusion_layer.state_dict(), os.path.join(save_dir, "hourly_fusion.pt"))
        # Also save the (possibly finetuned) upper kronos layers
        upper_state = {}
        for i in range(self.split_point, self.n_layers):
            prefix = f"transformer.{i}."
            for name, param in self.kronos.transformer[i].named_parameters():
                upper_state[prefix + name] = param.data
        for name, param in self.kronos.norm.named_parameters():
            upper_state["norm." + name] = param.data
        for name, param in self.kronos.dep_layer.named_parameters():
            upper_state["dep_layer." + name] = param.data
        for name, param in self.kronos.head.named_parameters():
            upper_state["head." + name] = param.data
        torch.save(upper_state, os.path.join(save_dir, "kronos_upper.pt"))

    def save_all(self, save_dir: str):
        """Save the full kronos predictor + C modules."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.kronos.save_pretrained(os.path.join(save_dir, "kronos_predictor"))
        self.save_c_modules(save_dir)

    @classmethod
    def load_for_inference(
        cls,
        kronos_predictor_path: str,
        c_modules_dir: str,
        hourly_encoder_kwargs: dict,
        fusion_kwargs: dict,
        split_point: int,
        device: str = "cpu",
    ) -> "KronosWithHourly":
        """Load a trained C-group model for inference."""
        kronos = Kronos.from_pretrained(kronos_predictor_path)

        encoder = HourlyEncoder(**hourly_encoder_kwargs)
        encoder.load_state_dict(torch.load(
            f"{c_modules_dir}/hourly_encoder.pt", map_location="cpu"
        ))

        fusion = HourlyFusionLayer(**fusion_kwargs)
        fusion.load_state_dict(torch.load(
            f"{c_modules_dir}/hourly_fusion.pt", map_location="cpu"
        ))

        # Load finetuned upper layers into kronos
        upper_state = torch.load(
            f"{c_modules_dir}/kronos_upper.pt", map_location="cpu"
        )
        kronos_state = kronos.state_dict()
        kronos_state.update(upper_state)
        kronos.load_state_dict(kronos_state)

        model = cls(kronos, encoder, fusion, split_point)
        model.eval().to(device)
        return model
