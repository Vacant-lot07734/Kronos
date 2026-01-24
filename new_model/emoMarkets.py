import sys
from tqdm import trange
sys.path.append("../")
from model.module import *

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

class MarketSpecificExpert(nn.Module):
    """针对不同市场的专家网络"""

    def __init__(self, emotion_dim=15, d_model=256, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        # 输入维度应为256，即卷积输出的维度
        self.emotion_processor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model)
        )

        # 修改这里：字段注意力应该作用在256维的卷积特征上，而不是原始15维情绪数据
        self.field_attn = nn.Linear(emotion_dim, emotion_dim)

        # 分组卷积层（使用合适的padding保持序列长度）
        self.index_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),  # 保持长度
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # 保持长度
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),  # 保持长度
            nn.ReLU()
        )
        self.limit_conv = nn.Sequential(
            nn.Conv1d(11, 64, kernel_size=7, padding=3),  # 保持长度
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # 保持长度
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),  # 保持长度
            nn.ReLU()
        )

    def forward(self, emotion_data):
        B, T, _ = emotion_data.shape

        # 字段注意力加权 - 现在作用在256维特征上
        field_scores = self.field_attn(emotion_data)
        field_weights = F.softmax(field_scores, dim=-1)
        emotion_weighted = emotion_data * field_weights

        # 分组卷积提取特征（注意维度变换）
        index_feat = self.index_conv(emotion_weighted[..., :4].transpose(1, 2)).transpose(1, 2)
        limit_feat = self.limit_conv(emotion_weighted[..., 4:].transpose(1, 2)).transpose(1, 2)
        emotion_conv = torch.cat([index_feat, limit_feat], dim=-1)  # (B, T, 256)

        # 处理并返回结果
        return self.emotion_processor(emotion_conv)


class MarketMoE(nn.Module):
    """市场特定的MoE模块"""

    def __init__(self, num_experts=4, emotion_dim=15, d_model=256):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model

        # 创建4个市场专家
        self.experts = nn.ModuleList([
            MarketSpecificExpert(emotion_dim, d_model, i)
            for i in range(num_experts)
        ])

        # 门控网络 - 根据情绪数据决定专家权重
        self.gate = nn.Sequential(
            nn.Linear(emotion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
        )

    def forward(self, emotion_data, market_type=None):
        """
        emotion_data: (B, T, 15)
        market_type: (B, T) - 市场类型标识 (0-3分别代表不同市场)
        """
        B, T, _ = emotion_data.shape

        # 如果提供了市场类型，则直接使用硬分配
        if market_type is not None:
            # 确保market_type在合理范围内
            market_type = torch.clamp(market_type, 0, self.num_experts - 1)
            gate_weights = F.one_hot(market_type, num_classes=self.num_experts).float()  # (B, T, num_experts)
        else:
            # 使用门控网络软分配
            gate_logits = self.gate(emotion_data)  # (B, T, num_experts)
            gate_weights = F.softmax(gate_logits, dim=-1)  # (B, T, num_experts)

        # 各专家处理
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(emotion_data)
            expert_outputs.append(expert_output)

        # 强制对齐所有张量的序列长度
        target_length = T  # 使用原始输入长度作为目标

        # 调整专家输出长度
        aligned_expert_outputs = []
        for expert_output in expert_outputs:
            current_length = expert_output.size(1)
            if current_length > target_length:
                # 截断
                expert_output = expert_output[:, :target_length, :]
            elif current_length < target_length:
                # 重复最后一个时间步或填充
                repeat_times = target_length - current_length
                last_frame = expert_output[:, -1:, :].repeat(1, repeat_times, 1)
                expert_output = torch.cat([expert_output, last_frame], dim=1)
            aligned_expert_outputs.append(expert_output)

        # 调整门控权重长度（确保正确处理维度）
        if gate_weights.dim() == 2:
            # 如果gate_weights是二维的，需要增加一个维度
            gate_weights = gate_weights.unsqueeze(-1)

        gate_length = gate_weights.size(1)
        if gate_length > target_length:
            gate_weights = gate_weights[:, :target_length, :]
        elif gate_length < target_length:
            repeat_times = target_length - gate_length
            if gate_weights.dim() == 3:
                last_frame = gate_weights[:, -1:, :].repeat(1, repeat_times, 1)
            else:
                last_frame = gate_weights[:, -1:].repeat(1, repeat_times)
            gate_weights = torch.cat([gate_weights, last_frame], dim=1)

        # 堆叠专家输出: (B, T, num_experts, d_model)
        expert_outputs = torch.stack(aligned_expert_outputs, dim=-2)

        # 验证维度一致性后再进行融合
        assert expert_outputs.size(1) == gate_weights.size(1), \
            f"Length mismatch after alignment: expert_outputs {expert_outputs.size(1)} vs gate_weights {gate_weights.size(1)}"

        # 加权融合: (B, T, d_model)
        gated_output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=-2)

        return gated_output, gate_weights


class MoEEmotionEnhancedTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    基于MoE的情绪增强型分词器，为不同市场提供定制化处理
    K线和情绪数据处理完全独立
    """

    def __init__(self, d_in: int, d_model: int, n_heads, ff_dim, n_enc_layers, n_dec_layers,
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p,
                 # K线量化器完整参数集
                 kline_s1_bits, kline_s2_bits,
                 kline_beta, kline_gamma0, kline_gamma, kline_zeta, kline_group_size,
                 # 情绪量化器完整参数集
                 emotion_s1_bits, emotion_s2_bits,
                 emotion_beta, emotion_gamma0, emotion_gamma, emotion_zeta, emotion_group_size,
                 emotion_dim=15):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.emotion_dim = emotion_dim

        # K线量化相关参数（完全独立）
        self.kline_s1_bits = kline_s1_bits
        self.kline_s2_bits = kline_s2_bits
        self.kline_codebook_dim: int = kline_s1_bits + kline_s2_bits
        self.kline_embed = nn.Linear(self.d_in, self.d_model)
        self.head_kline = nn.Linear(self.d_model, self.d_in)

        # K线独立编码器/解码器
        self.encoder_kline = nn.ModuleList([
            self._create_transformer_block()
            for _ in range(self.enc_layers)
        ])
        self.decoder_kline = nn.ModuleList([
            self._create_transformer_block()
            for _ in range(self.dec_layers)
        ])

        # K线量化前后投影（使用K线独立参数）
        self.quant_embed_kline = nn.Linear(self.d_model, self.kline_codebook_dim)
        self.kline_post_quant_embed_pre = nn.Linear(self.kline_s1_bits, self.d_model)
        self.kline_post_quant_embed_full = nn.Linear(self.kline_codebook_dim, self.d_model)

        from model.module import BSQuantizer
        # K线独立量化器
        self.kline_tokenizer = BSQuantizer(self.kline_s1_bits, self.kline_s2_bits,
                                           kline_beta, kline_gamma0, kline_gamma, kline_zeta, kline_group_size)

        # 情绪处理模块（完全独立）
        self.emotion_s1_bits = emotion_s1_bits
        self.emotion_s2_bits = emotion_s2_bits
        self.emotion_codebook_dim: int = emotion_s1_bits + emotion_s2_bits
        self.emotion_embed = nn.Linear(emotion_dim, d_model)
        self.encoder_emotion = nn.ModuleList([
            self._create_transformer_block()
            for _ in range(self.enc_layers)
        ])
        self.decoder_emotion = nn.ModuleList([
            self._create_transformer_block()
            for _ in range(self.dec_layers)
        ])

        # 修正情绪量化相关组件（使用情绪独立参数）
        self.emotion_quant_embed = nn.Linear(self.d_model, self.emotion_codebook_dim)  # 修正：使用情绪码本维度
        self.emotion_quant_embed_pre = nn.Linear(self.d_model, self.emotion_s1_bits)  # 新增：情绪s1量化
        self.emotion_post_quant_embed_pre = nn.Linear(self.emotion_s1_bits, self.d_model)  # 新增：情绪s1重构
        self.emotion_post_quant_embed_full = nn.Linear(self.emotion_codebook_dim, self.d_model)  # 修正：使用情绪码本维度
        self.head_emotion = nn.Linear(self.d_model, emotion_dim)

        # 情绪独立量化器
        self.emotion_tokenizer = BSQuantizer(self.emotion_s1_bits, self.emotion_s2_bits,  # 只有s1层
                                                emotion_beta, emotion_gamma0, emotion_gamma, emotion_zeta,
                                                emotion_group_size)

    def _create_transformer_block(self):
        from model.module import TransformerBlock
        return TransformerBlock(
            self.d_model,
            self.n_heads,
            self.ff_dim,
            self.ffn_dropout_p,
            self.attn_dropout_p,
            self.resid_dropout_p
        )

    def _process_emotion_features(self, emotion):
        """
        采用与K线相同的处理逻辑生成情绪token（使用独立量化器和参数）
        """
        if emotion is not None:
            # 与K线处理方式统一：嵌入 -> Transformer编码 -> 量化
            z_emotion = self.emotion_embed(emotion)  # emotion_dim -> d_model
            for layer in self.encoder_emotion:
                z_emotion = layer(z_emotion)

            z_emotion = self.emotion_quant_embed(z_emotion)
            bsq_loss_emotion, quantized_emotion, emotion_tokens = self.emotion_tokenizer(z_emotion)

            quantized_emotion_pre = quantized_emotion[:,:,:self.emotion_s1_bits]

            z_emotion_pre = self.emotion_post_quant_embed_pre(quantized_emotion_pre)

            z_emotion = self.emotion_post_quant_embed_full(quantized_emotion)
            for layer in self.decoder_emotion:
                z_emotion_pre = layer(z_emotion_pre)
            z_emotion_pre = self.head_emotion(z_emotion_pre)
            for layer in self.decoder_emotion:
                z_emotion = layer(z_emotion)

            z_emotion = self.head_emotion(z_emotion)
            return (z_emotion_pre,z_emotion),bsq_loss_emotion,quantized_emotion,emotion_tokens
        return None

    def forward(self, x, emotion=None):
        """
        前向传播 - K线和情绪数据使用完全独立的参数
        """
        # K线独立编码与量化
        z_kline = self.kline_embed(x)
        for layer in self.encoder_kline:
            z_kline = layer(z_kline)

        z_kline = self.quant_embed_kline(z_kline)
        kline_bsq_loss, quantized_kline, z_indices = self.kline_tokenizer(z_kline)

        # K线分层重构（使用K线独立投影层）
        quant_pre = quantized_kline[..., :self.kline_s1_bits]
        z_pre = self.kline_post_quant_embed_pre(quant_pre)
        z_full = self.kline_post_quant_embed_full(quantized_kline)
        for layer in self.decoder_kline:
            z_pre = layer(z_pre)
        z_pre = self.head_kline(z_pre)



        for layer in self.decoder_kline:
            z_full = layer(z_full)
        z_full = self.head_kline(z_full)




        (z_emotion_pre,z_emotion),bsq_loss_emotion,quantized_emotion,emotion_tokens = self._process_emotion_features(emotion)


        return (z_pre, z_full),kline_bsq_loss,quantized_kline,z_indices,(z_emotion_pre,z_emotion),bsq_loss_emotion,quantized_emotion,emotion_tokens

    def indices_to_bits(self, x, half=False):
        """索引转比特表示（支持不同模态）"""
        # 根据x的形状推断是K线还是情绪数据
        if half:
            x1 = x[0]  # Assuming x is a tuple of indices if half is True
            x2 = x[1]

            # 根据张量大小判断是K线还是情绪码本维度
            if x1.size(-1) == self.kline_s1_bits + self.kline_s2_bits:
                codebook_dim = self.kline_codebook_dim
            elif x1.size(-1) == self.emotion_s1_bits + self.emotion_s2_bits:
                codebook_dim = self.emotion_codebook_dim
            else:
                # 如果都不匹配，可以根据默认值选择
                codebook_dim = self.kline_codebook_dim  # 默认使用K线码本维度

            mask = 2 ** torch.arange(codebook_dim // 2, device=x1.device,
                                     dtype=torch.long)  # Create a mask for bit extraction
            x1 = (x1.unsqueeze(-1) & mask) != 0  # Extract bits for the first half
            x2 = (x2.unsqueeze(-1) & mask) != 0  # Extract bits for the second half
            x = torch.cat([x1, x2], dim=-1)  # Concatenate the bit representations
        else:
            # 根据张量大小判断是K线还是情绪码本维度
            if x.size(-1) == self.kline_codebook_dim:
                codebook_dim = self.kline_codebook_dim
            elif x.size(-1) == self.emotion_codebook_dim:
                codebook_dim = self.emotion_codebook_dim
            else:
                # 如果都不匹配，可以根据默认值选择
                codebook_dim = self.kline_codebook_dim  # 默认使用K线码本维度

            mask = 2 ** torch.arange(codebook_dim, device=x.device,
                                     dtype=torch.long)  # Create a mask for bit extraction
            x = (x.unsqueeze(-1) & mask) != 0  # Extract bits

        x = x.float() * 2 - 1  # Convert boolean to bipolar (-1, 1)
        q_scale = 1. / (codebook_dim ** 0.5)  # Scaling factor
        x = x * q_scale
        return x

    def encode(self, x, emotion=None, market_type=None, half=False):
        """
        编码接口 - 独立编码K线和情绪数据
        """
        # K线独立编码
        z_kline = self.kline_embed(x)
        for layer in self.encoder_kline:
            z_kline = layer(z_kline)
        z_kline = self.quant_embed_kline(z_kline)
        kline_bsq_loss, quantized_kline, z_indices = self.kline_tokenizer(z_kline,half)

        # 情绪独立编码
        z_emotion = self.emotion_embed(emotion)  # emotion_dim -> d_model
        for layer in self.encoder_emotion:
            z_emotion = layer(z_emotion)

        z_emotion = self.emotion_quant_embed(z_emotion)

        bsq_loss_emotion, quantized_emotion, emotion_tokens = self.emotion_tokenizer(z_emotion,half)

        return z_indices, emotion_tokens

    def decode(self, x, emotion, half=False):
        """解码（支持不同模态）"""
        quantized_kline = self.indices_to_bits(x,  half)
        quantized_emotion = self.indices_to_bits(emotion,  half)
        z_full = self.kline_post_quant_embed_full(quantized_kline)
        for layer in self.decoder_kline:
            z_full = layer(z_full)
        z_full = self.head_kline(z_full)
        z_emotion = self.emotion_post_quant_embed_full(quantized_emotion)
        for layer in self.decoder_emotion:
            z_emotion = layer(z_emotion)
        z_emotion = self.head_emotion(z_emotion)
        return z_full, z_emotion



def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = torch.topk(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x


def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip=5, T=1.0, top_k=None,
                              top_p=0.99, sample_count=5, verbose=False, emotion=None, market_indices=None):
    with torch.no_grad():
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(
            device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(
            device)

        # 处理市场索引
        if market_indices is not None:
            market_indices = market_indices.unsqueeze(1).repeat(1, sample_count, 1).reshape(-1).to(device)

        # 处理情绪数据
        emotion_tensor = None
        if emotion is not None:
            emotion = emotion.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, emotion.size(1),
                                                                                 emotion.size(2)).to(device)
            emotion_tensor = emotion

        # 获取情绪tokens - 根据新模型参数调整
        # 获取分层情绪tokens
        encode_result = tokenizer.encode(x, emotion_tensor, market_indices, half=True)
        if isinstance(encode_result, tuple):
            x_token, emotion_tokens = encode_result
            # 分离粗粒度和细粒度情绪tokens
            if isinstance(emotion_tokens, tuple):
                emotion_coarse_tokens, emotion_fine_tokens = emotion_tokens
            else:
                emotion_coarse_tokens = emotion_tokens
                emotion_fine_tokens = None
        else:
            x_token = encode_result
            emotion_coarse_tokens = None
            emotion_fine_tokens = None

        emotion_scores = None
        if emotion_tensor is not None:
            emotion_scores = model.compute_emotion_intensity(emotion_tensor)

        def get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, pred_step):
            if current_seq_len <= max_context - pred_step:
                return torch.cat([x_stamp, y_stamp[:, :pred_step, :]], dim=1)
            else:
                start_idx = max_context - pred_step
                return torch.cat([x_stamp[:, -start_idx:, :], y_stamp[:, :pred_step, :]], dim=1)

        if verbose:
            ran = trange
        else:
            ran = range

        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i

            if current_seq_len <= max_context:
                input_tokens = x_token
            else:
                input_tokens = [t[:, -max_context:].contiguous() for t in x_token]

            current_stamp = get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, i)

            s1_logits, context = model.decode_s1(
                input_tokens[0],
                input_tokens[1],
                current_stamp,
                emotion_coarse_tokens=emotion_coarse_tokens,  # 修改：使用新的参数名称
                emotion_fine_tokens=emotion_fine_tokens,     # 修改：使用新的参数名称
                emotion_scores=emotion_scores,
                kline_features=x,
                market_indices=market_indices
            )

            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(
                context,
                sample_pre,
                emotion_coarse_tokens=emotion_coarse_tokens,  # 修改：使用新的参数名称
                emotion_fine_tokens=emotion_fine_tokens     # 修改：使用新的参数名称
            )
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            x_token[0] = torch.cat([x_token[0], sample_pre], dim=1)
            x_token[1] = torch.cat([x_token[1], sample_post], dim=1)

            torch.cuda.empty_cache()

        input_tokens = [t[:, -max_context:].contiguous() for t in x_token]

        # 修复：处理解码器返回的元组，只取K线部分
        z = tokenizer.decode(input_tokens, emotion_tokens, half=True)
        if isinstance(z, tuple):
            # 如果返回元组，只取第一个元素（K线解码结果）
            z = z[0]

        z = z.reshape(batch_size, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)

        return preds




def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()

    if isinstance(x_timestamp, pd.DatetimeIndex):
        time_df['minute'] = x_timestamp.minute
        time_df['hour'] = x_timestamp.hour
        time_df['weekday'] = x_timestamp.weekday
        time_df['day'] = x_timestamp.day
        time_df['month'] = x_timestamp.month
    else:
        time_df['minute'] = x_timestamp.dt.minute
        time_df['hour'] = x_timestamp.dt.hour
        time_df['weekday'] = x_timestamp.dt.weekday
        time_df['day'] = x_timestamp.dt.day
        time_df['month'] = x_timestamp.dt.month

    time_df['minute'] = time_df['minute'].fillna(0)
    time_df['hour'] = time_df['hour'].fillna(0)

    return time_df


class EmotionEnhancedWithMoE(nn.Module, PyTorchModelHubMixin):
    """
    使用MoE的情绪增强版Kronos模型，支持市场特定处理
    """

    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim,
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te, num_markets=4):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p
        self.num_markets = num_markets

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)

        # K线嵌入
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)

        # **新增：情绪token的独立嵌入层**
        self.emotion_embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)

        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p,
                             self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

        # 添加跨模态注意力机制
        self.cross_attention_emotion_to_kline = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)
        self.cross_attention_kline_to_emotion = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)
        # 多模态融合后的投影层
        self.fusion_projection = nn.Linear(d_model, d_model)

        # 动态权重网络
        self.dynamic_weight_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 市场特定专家网络 (MoE)
        self.market_moe = MarketMoE(num_experts=num_markets, emotion_dim=15, d_model=d_model)

        # 市场嵌入
        self.market_embedding = nn.Embedding(num_markets, 32)
        self.market_fusion = nn.Linear(d_model + 32, d_model)

        # K线处理器
        self.kline_processor = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p,
                             self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])

        # **新增：情绪处理器**
        self.emotion_processor = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p,
                             self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])

        # 可学习的情绪强度计算参数
        self.emotion_index_weights = nn.Parameter(torch.ones(4))
        self.emotion_limit_weights = nn.Parameter(torch.ones(11))
        self.emotion_combined_weights = nn.Parameter(torch.ones(2))

    def compute_kline_derived_indices(self, batch_x):
        """
        从原始K线计算衍生指标（适配A股日频数据）
        Args:
            batch_x: 原始K线特征 (B, T, d_in=6)，字段顺序：[open, high, low, close, volume, amount]
        Returns:
            r_close: 收盘价涨幅 (B, T)
            r_volume: 成交量变化率 (B, T)
        """
        close = batch_x[..., 3]  # 第4维：收盘价
        volume = batch_x[..., 4]  # 第5维：成交量

        # 1. 收盘价涨幅：(当前收盘价 - 前一时刻收盘价) / 前一时刻收盘价
        r_close = close.diff(dim=1) / (close[:, :-1] + 1e-8)  # 添加小量避免除零
        # 填充首个时刻的NaN（设为0）
        r_close = torch.cat([torch.zeros(batch_x.size(0), 1, device=batch_x.device), r_close], dim=1)  # (B, T)

        # 2. 成交量变化率：(当前成交量 - 前一时刻成交量) / 前一时刻成交量
        r_volume = volume.diff(dim=1) / (volume[:, :-1] + 1e-8)  # 添加小量避免除零
        r_volume = torch.cat([torch.zeros(batch_x.size(0), 1, device=batch_x.device), r_volume], dim=1)  # (B, T)

        # 限制范围
        r_close = r_close.clip(min=-0.2, max=0.2)  # 涨跌停限制±20%
        r_volume = r_volume.clip(min=-0.9, max=10.0)  # 成交量波动放宽限制

        return r_close, r_volume

    def compute_emotion_intensity(self, emotion_data):
        """
        使用可学习参数计算情绪强度
        """
        if isinstance(emotion_data, np.ndarray):
            # 归一化权重
            index_weights = F.softmax(self.emotion_index_weights, dim=0)
            limit_weights = F.softmax(self.emotion_limit_weights, dim=0)
            combined_weights = F.softmax(self.emotion_combined_weights, dim=0)

            e_index = (index_weights[0] * emotion_data[..., 0] +
                       index_weights[1] * emotion_data[..., 1] +
                       index_weights[2] * emotion_data[..., 2] +
                       index_weights[3] * emotion_data[..., 3])

            e_limit = (limit_weights[0] * emotion_data[..., 4] +
                       limit_weights[1] * emotion_data[..., 5] +
                       limit_weights[2] * emotion_data[..., 6] +
                       limit_weights[3] * emotion_data[..., 7] +
                       limit_weights[4] * emotion_data[..., 8] +
                       limit_weights[5] * emotion_data[..., 9] +
                       limit_weights[6] * emotion_data[..., 10] +
                       limit_weights[7] * emotion_data[..., 11] +
                       limit_weights[8] * emotion_data[..., 12] +
                       limit_weights[9] * emotion_data[..., 13] +
                       limit_weights[10] * emotion_data[..., 14])

            e_limit = np.clip(e_limit, 0, 1)
            e_d = combined_weights[0] * e_index + combined_weights[1] * e_limit
        else:
            # 归一化权重
            index_weights = F.softmax(self.emotion_index_weights, dim=0)
            limit_weights = F.softmax(self.emotion_limit_weights, dim=0)
            combined_weights = F.softmax(self.emotion_combined_weights, dim=0)

            e_index = (index_weights[0] * emotion_data[..., 0] +
                       index_weights[1] * emotion_data[..., 1] +
                       index_weights[2] * emotion_data[..., 2] +
                       index_weights[3] * emotion_data[..., 3])

            e_limit = (limit_weights[0] * emotion_data[..., 4] +
                       limit_weights[1] * emotion_data[..., 5] +
                       limit_weights[2] * emotion_data[..., 6] +
                       limit_weights[3] * emotion_data[..., 7] +
                       limit_weights[4] * emotion_data[..., 8] +
                       limit_weights[5] * emotion_data[..., 9] +
                       limit_weights[6] * emotion_data[..., 10] +
                       limit_weights[7] * emotion_data[..., 11] +
                       limit_weights[8] * emotion_data[..., 12] +
                       limit_weights[9] * emotion_data[..., 13] +
                       limit_weights[10] * emotion_data[..., 14])

            e_limit = torch.clamp(e_limit, 0, 1)
            e_d = combined_weights[0] * e_index + combined_weights[1] * e_limit

        return e_d

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def get_dynamic_weights(self, emotion_scores, kline_features):
        volatility = torch.std(kline_features[..., 3], dim=1, keepdim=True)
        volume_mean = torch.mean(kline_features[..., 4], dim=1, keepdim=True)

        if emotion_scores.dim() == 1:
            emotion_scores_expanded = emotion_scores.unsqueeze(0).unsqueeze(-1)
        elif emotion_scores.dim() == 2:
            if emotion_scores.size(1) == 1:
                emotion_scores_expanded = emotion_scores.unsqueeze(1).expand(-1, kline_features.size(1), -1)
            else:
                emotion_scores_expanded = emotion_scores.unsqueeze(-1)
        else:
            emotion_scores_expanded = emotion_scores

        batch_size = kline_features.size(0)
        seq_len = kline_features.size(1)

        if volatility.dim() == 1:
            volatility_expanded = volatility.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, -1)
        elif volatility.dim() == 2:
            if volatility.size(1) == 1:
                volatility_expanded = volatility.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                volatility_expanded = volatility.unsqueeze(-1)
        else:
            volatility_expanded = volatility

        if volume_mean.dim() == 1:
            volume_mean_expanded = volume_mean.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, -1)
        elif volume_mean.dim() == 2:
            if volume_mean.size(1) == 1:
                volume_mean_expanded = volume_mean.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                volume_mean_expanded = volume_mean.unsqueeze(-1)
        else:
            volume_mean_expanded = volume_mean

        if emotion_scores_expanded.dim() != 3:
            if emotion_scores_expanded.dim() == 2:
                emotion_scores_expanded = emotion_scores_expanded.unsqueeze(-1)

        if volatility_expanded.dim() != 3:
            if volatility_expanded.dim() == 2:
                volatility_expanded = volatility_expanded.unsqueeze(-1)

        if volume_mean_expanded.dim() != 3:
            if volume_mean_expanded.dim() == 2:
                volume_mean_expanded = volume_mean_expanded.unsqueeze(-1)

        combined_features = torch.cat([emotion_scores_expanded, volatility_expanded, volume_mean_expanded], dim=-1)

        w_emotion = self.dynamic_weight_net(combined_features) * 0.8 + 0.1
        w_kline = 1.0 - w_emotion

        if w_emotion.dim() > 2:
            w_emotion = w_emotion.squeeze(-1)
        if w_kline.dim() > 2:
            w_kline = w_kline.squeeze(-1)

        return w_emotion, w_kline

    def process_emotion_features(self, emotion_tokens, market_indices=None):
        """
        处理情绪特征，支持市场特定处理
        """
        # 检查是否已经是处理过的特征（256维）还是原始情绪数据（15维）
        if emotion_tokens.size(-1) == 15:
            # 如果是原始情绪数据，则使用MoE处理
            if market_indices is not None:
                emotion_features, gate_weights = self.market_moe(emotion_tokens, market_indices)
            else:
                emotion_features = emotion_tokens
                gate_weights = None
        else:
            # 如果已经是处理过的特征，直接使用
            emotion_features = emotion_tokens
            gate_weights = None

        # 融合市场嵌入信息（如果需要且输入是原始情绪数据）
        if market_indices is not None and emotion_features.size(-1) == 15:
            # 融合市场嵌入信息
            if market_indices.dim() == 1:
                market_indices = market_indices.unsqueeze(-1)

            if market_indices.dim() == 2:
                if market_indices.size(1) == 1:
                    market_indices = market_indices.expand(-1, emotion_features.size(1))
                elif market_indices.size(1) != emotion_features.size(1):
                    target_len = emotion_features.size(1)
                    if market_indices.size(1) > target_len:
                        market_indices = market_indices[:, :target_len]
                    else:
                        pad_len = target_len - market_indices.size(1)
                        last_vals = market_indices[:, -1:].repeat(1, pad_len)
                        market_indices = torch.cat([market_indices, last_vals], dim=1)

            market_indices = market_indices.long()
            market_embed = self.market_embedding(market_indices)
            fused_features = torch.cat([emotion_features, market_embed], dim=-1)
            emotion_features = self.market_fusion(fused_features)

        return emotion_features, gate_weights

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False,
                s1_targets=None, emotion_coarse_tokens=None, emotion_fine_tokens=None,
                emotion_scores=None, kline_features=None, market_indices=None,
                original_emotion=None):

        # K线token处理 - 完全独立的处理流程
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        # 对K线token进行处理
        for layer in self.kline_processor:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        kline_tokens = x

        # 情绪token处理 - 使用与K线完全相同的处理流程
        x_emotion = self.emotion_embedding([emotion_coarse_tokens, emotion_fine_tokens])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x_emotion = x_emotion + time_embedding
        x_emotion = self.token_drop(x_emotion)
        for layer in self.emotion_processor:
            x_emotion = layer(x_emotion, key_padding_mask=padding_mask)
        x_emotion = self.norm(x_emotion)
        emotion_tokens = x_emotion

        # 跨模态融合
        if emotion_tokens is not None and emotion_scores is not None:
            # 使用市场特定处理
            if market_indices is not None:
                emotion_tokens, _ = self.process_emotion_features(emotion_tokens, market_indices)

            # 计算情绪强度（如果还没有计算）
            if emotion_scores is None and emotion_tokens is not None:
                emotion_scores = self.compute_emotion_intensity(emotion_tokens)

            # 交叉注意力机制
            cross_attention_e2k = self.cross_attention_emotion_to_kline(
                query=emotion_tokens,
                key=kline_tokens,
                value=kline_tokens,
                key_padding_mask=padding_mask
            )

            cross_attention_k2e = self.cross_attention_kline_to_emotion(
                query=kline_tokens,
                key=emotion_tokens,
                value=emotion_tokens,
                key_padding_mask=padding_mask
            )

            w_emotion, w_kline = self.get_dynamic_weights(emotion_scores,
                                                          kline_features) if kline_features is not None else (None,
                                                                                                              None)

            if w_emotion is not None and w_kline is not None:
                if w_emotion.dim() == 1:
                    w_emotion = w_emotion.unsqueeze(0)
                if w_emotion.dim() == 2 and w_emotion.size(0) == 1 and emotion_tokens.size(0) > 1:
                    w_emotion = w_emotion.expand(emotion_tokens.size(0), -1)

                if w_emotion.dim() == 2:
                    w_emotion = w_emotion.unsqueeze(-1)

                if w_emotion.size(-1) == 1 and emotion_tokens.size(-1) > 1:
                    w_emotion = w_emotion.expand(-1, -1, emotion_tokens.size(-1))

                if w_kline.dim() == 1:
                    w_kline = w_kline.unsqueeze(0)
                if w_kline.dim() == 2 and w_kline.size(0) == 1 and kline_tokens.size(0) > 1:
                    w_kline = w_kline.expand(kline_tokens.size(0), -1)

                if w_kline.dim() == 2:
                    w_kline = w_kline.unsqueeze(-1)

                if w_kline.size(-1) == 1 and kline_tokens.size(-1) > 1:
                    w_kline = w_kline.expand(-1, -1, kline_tokens.size(-1))

                x = cross_attention_e2k + cross_attention_k2e + w_emotion * emotion_tokens + w_kline * kline_tokens
            else:
                x = cross_attention_e2k + cross_attention_k2e + emotion_tokens + kline_tokens

        # 通过投影层处理融合后的特征
        x = self.fusion_projection(x)
        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)

        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None,
                  emotion_coarse_tokens=None, emotion_fine_tokens=None,
                  emotion_scores=None, kline_features=None, market_indices=None):

        # K线token处理 - 完全独立的处理流程
        x = self.embedding([s1_ids, s2_ids])

        if stamp is not None:
            if stamp.size(1) != s1_ids.size(1):
                raise ValueError(
                    f"Timestamp length {stamp.size(1)} doesn't match token sequence length {s1_ids.size(1)}")
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding

        x = self.token_drop(x)

        # 对K线token进行处理
        for layer in self.kline_processor:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        kline_tokens = x

        # 情绪token处理 - 使用与K线完全相同的处理流程
        x_emotion = None
        emotion_tokens = None

        if emotion_coarse_tokens is not None and emotion_fine_tokens is not None:
            # 确保情绪tokens是正确的tensor类型
            if isinstance(emotion_coarse_tokens, list):
                emotion_coarse_tokens = torch.stack(emotion_coarse_tokens) if len(emotion_coarse_tokens) > 0 else None
            if isinstance(emotion_fine_tokens, list):
                emotion_fine_tokens = torch.stack(emotion_fine_tokens) if len(emotion_fine_tokens) > 0 else None

            if emotion_coarse_tokens is not None and emotion_fine_tokens is not None:
                # 确保是tensor类型
                if not isinstance(emotion_coarse_tokens, torch.Tensor):
                    emotion_coarse_tokens = torch.tensor(emotion_coarse_tokens, dtype=torch.long)
                if not isinstance(emotion_fine_tokens, torch.Tensor):
                    emotion_fine_tokens = torch.tensor(emotion_fine_tokens, dtype=torch.long)

                # 确保维度一致
                if emotion_coarse_tokens.dim() == 1:
                    emotion_coarse_tokens = emotion_coarse_tokens.unsqueeze(0)
                if emotion_fine_tokens.dim() == 1:
                    emotion_fine_tokens = emotion_fine_tokens.unsqueeze(0)

                x_emotion = self.emotion_embedding([emotion_coarse_tokens, emotion_fine_tokens])

                if stamp is not None:
                    time_embedding = self.time_emb(stamp)
                    x_emotion = x_emotion + time_embedding
                x_emotion = self.token_drop(x_emotion)
                for layer in self.emotion_processor:
                    x_emotion = layer(x_emotion, key_padding_mask=padding_mask)
                x_emotion = self.norm(x_emotion)
                emotion_tokens = x_emotion

        if emotion_tokens is not None and emotion_scores is not None:
            # 使用市场特定处理
            if market_indices is not None:
                emotion_tokens, _ = self.process_emotion_features(emotion_tokens, market_indices)

            # 计算情绪强度（如果还没有计算）
            if emotion_scores is None and emotion_tokens is not None:
                emotion_scores = self.compute_emotion_intensity(emotion_tokens)

            # 交叉注意力机制
            cross_attention_e2k = self.cross_attention_emotion_to_kline(
                query=emotion_tokens,
                key=kline_tokens,
                value=kline_tokens,
                key_padding_mask=padding_mask
            )

            cross_attention_k2e = self.cross_attention_kline_to_emotion(
                query=kline_tokens,
                key=emotion_tokens,
                value=emotion_tokens,
                key_padding_mask=padding_mask
            )

            w_emotion, w_kline = self.get_dynamic_weights(emotion_scores,
                                                          kline_features) if kline_features is not None else (None,
                                                                                                              None)

            if w_emotion is not None and w_kline is not None:
                if w_emotion.dim() == 1:
                    w_emotion = w_emotion.unsqueeze(0)
                if w_emotion.dim() == 2 and w_emotion.size(0) == 1 and emotion_tokens.size(0) > 1:
                    w_emotion = w_emotion.expand(emotion_tokens.size(0), -1)

                if w_emotion.dim() == 2:
                    w_emotion = w_emotion.unsqueeze(-1)

                if w_emotion.size(-1) == 1 and emotion_tokens.size(-1) > 1:
                    w_emotion = w_emotion.expand(-1, -1, emotion_tokens.size(-1))

                if w_kline.dim() == 1:
                    w_kline = w_kline.unsqueeze(0)
                if w_kline.dim() == 2 and w_kline.size(0) == 1 and kline_tokens.size(0) > 1:
                    w_kline = w_kline.expand(kline_tokens.size(0), -1)

                if w_kline.dim() == 2:
                    w_kline = w_kline.unsqueeze(-1)

                if w_kline.size(-1) == 1 and kline_tokens.size(-1) > 1:
                    w_kline = w_kline.expand(-1, -1, kline_tokens.size(-1))

                x = cross_attention_e2k + cross_attention_k2e + w_emotion * emotion_tokens + w_kline * kline_tokens
            else:
                x = cross_attention_e2k + cross_attention_k2e + emotion_tokens + kline_tokens

        else:
            # 当没有情绪数据时，只使用K线数据
            x = kline_tokens

        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None,
                  emotion_coarse_tokens=None, emotion_fine_tokens=None):  # 修改参数名称

        sibling_embed = self.embedding.emb_s1(s1_ids)  # 使用统一的embedding
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)

        # 如果有情绪数据，也要考虑情绪嵌入
        if emotion_coarse_tokens is not None and emotion_fine_tokens is not None:
            # 确保情绪tokens是正确的tensor类型
            if isinstance(emotion_coarse_tokens, list):
                emotion_coarse_tokens = torch.stack(emotion_coarse_tokens) if len(emotion_coarse_tokens) > 0 else None
            if isinstance(emotion_fine_tokens, list):
                emotion_fine_tokens = torch.stack(emotion_fine_tokens) if len(emotion_fine_tokens) > 0 else None

            if emotion_coarse_tokens is not None and emotion_fine_tokens is not None:
                # 确保是tensor类型
                if not isinstance(emotion_coarse_tokens, torch.Tensor):
                    emotion_coarse_tokens = torch.tensor(emotion_coarse_tokens, dtype=torch.long)
                if not isinstance(emotion_fine_tokens, torch.Tensor):
                    emotion_fine_tokens = torch.tensor(emotion_fine_tokens, dtype=torch.long)

                emotion_embed = self.emotion_embedding([emotion_coarse_tokens, emotion_fine_tokens])
                x2 = x2 + emotion_embed  # 简单融合情绪信息

        return self.head.cond_forward(x2)


class EmoMarketsPredictorWithMoE:
    def __init__(self, model, tokenizer, device="cuda:0", max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'vol'
        self.amt_vol = 'amount'
        self.time_cols = ['weekday', 'day', 'month']
        self.device = device
        self.emotion_cols = ['index_return_sh', 'index_return_sz', 'index_return_cyb', 'index_return_bse',
                            'limit_up_1st', 'limit_up_2nd', 'limit_up_3rd', 'limit_up_4th',
                            'limit_up_5th', 'limit_up_6th', 'limit_down', 'limit_up_20cm_1st',
                            'limit_up_20cm', 'limit_up_30cm_1st', 'limit_up_30cm']

        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

        # 定义市场映射关系
        self.market_mapping = {
            'sh': 0,  # 上证
            'sz': 1,  # 深证
            'cyb': 2, # 创业板
            'bse': 3  # 北交所
        }

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count,
                 verbose, emotion=None, y_emotion=None, market_indices=None):
        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        if emotion is not None:
            emotion_tensor = torch.from_numpy(np.array(emotion).astype(np.float32)).to(self.device)
        else:
            emotion_tensor = None

        if market_indices is not None:
            market_indices_tensor = torch.from_numpy(np.array(market_indices)).to(self.device)
        else:
            market_indices_tensor = None

        preds = auto_regressive_inference(
            self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor,
            self.max_context, pred_len,
            self.clip, T, top_k, top_p, sample_count, verbose,
            emotion_tensor, market_indices_tensor
        )
        return preds

    def predict(self, df, x_timestamp, y_timestamp, pred_len, emotion_data=None,
                y_emotion_data=None, market_type=None, T=1.0, top_k=0,
                top_p=0.9, sample_count=1, verbose=True):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_vol] = 0.0
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        if emotion_data is not None:
            emotion = emotion_data[self.emotion_cols].values.astype(np.float32)
            emotion = emotion[np.newaxis, :]
        else:
            emotion = None

        if y_emotion_data is not None:
            y_emotion = y_emotion_data[self.emotion_cols].values.astype(np.float32)
            y_emotion = y_emotion[np.newaxis, :]
        else:
            y_emotion = None
        # 处理市场索引
        if market_type is not None and market_type in self.market_mapping:
            market_indices = np.full((1,), self.market_mapping[market_type])  # batch_size=1
        else:
            market_indices = None

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count,
                              verbose, emotion, y_emotion, market_indices)

        preds = preds.squeeze(0)
        # 关键修改：只取最后pred_len个预测结果
        preds = preds[-pred_len:, :]
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp)
        return pred_df

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, emotion_data_list=None,
                      market_types=None, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(
                y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []
        emotion_list = []
        market_indices_list = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(
                    f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(
                    f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

            if emotion_data_list is not None and i < len(emotion_data_list):
                emotion_df = emotion_data_list[i]
                emotion = emotion_df[self.emotion_cols].values.astype(np.float32)
                emotion_list.append(emotion)
            else:
                emotion_list.append(None)

            # 处理市场类型
            if market_types is not None and i < len(market_types):
                market_type = market_types[i]
                if market_type in self.market_mapping:
                    market_indices_list.append(self.market_mapping[market_type])
                else:
                    market_indices_list.append(0)  # 默认为上证
            else:
                market_indices_list.append(0)  # 默认为上证

        if len(set(seq_lens)) != 1:
            raise ValueError(
                f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(
                f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)
        market_indices_batch = np.array(market_indices_list)

        if any(emotion is not None for emotion in emotion_list):
            emotion_batch = np.stack(emotion_list, axis=0).astype(np.float32)
        else:
            emotion_batch = None

        # 转换市场索引为张量
        if market_indices_batch is not None:
            market_indices_tensor = torch.from_numpy(market_indices_batch).to(self.device)
        else:
            market_indices_tensor = None

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose,
                              emotion_batch, market_indices=market_indices_tensor)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(preds_i, columns=self.price_cols + [self.vol_col, self.amt_vol],
                                   index=y_timestamp_list[i])
            pred_dfs.append(pred_df)

        return pred_dfs
