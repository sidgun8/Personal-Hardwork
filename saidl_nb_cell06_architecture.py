# Cell 6 — Positional encodings + attention variants + model (SAiDL Core ML)
# Injected into SAIDL_BPGC_AttentionVariants_WandB.ipynb — keep in sync when editing.

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityPositionalEncoding(nn.Module):
    def forward(self, x):
        return x


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.pos_emb(pos)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


def _rotate_half(x):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE frequencies — applied inside attention to Q/K."""

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, q, k):
        """q,k: (B, H, T, Dh); Dh must be even."""
        B, H, T, Dh = q.shape
        t = torch.arange(T, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed


def causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


def build_alibi_slopes(n_heads: int, device, dtype):
    """Geometric slopes per head (Press et al., ALiBi)."""
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    slopes = torch.pow(2.0, -torch.arange(0, closest_power_of_2, dtype=dtype, device=device) / closest_power_of_2)
    if closest_power_of_2 != n_heads:
        extra = torch.pow(2.0, -torch.arange(1, 2 * (n_heads - closest_power_of_2) + 1, 2, dtype=dtype, device=device) / closest_power_of_2)
        slopes = torch.cat([slopes, extra], dim=0)
    slopes = slopes[:n_heads].view(n_heads, 1, 1)
    return slopes


def alibi_bias(n_heads: int, T: int, device, dtype):
    """Additive causal bias (H, T, T): -slope * (i-j) for j<=i."""
    slopes = build_alibi_slopes(n_heads, device, dtype)
    pos = torch.arange(T, device=device)
    dist = pos.unsqueeze(0) - pos.unsqueeze(1)
    causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    dist = dist.masked_fill(causal, 0)
    bias = -slopes * dist.unsqueeze(0).to(dtype)
    bias = bias.masked_fill(causal.unsqueeze(0), float("-inf"))
    return bias


class RelativePositionBias(nn.Module):
    """Learned bias buckets by relative distance (causal), Shaw-style simplified."""

    def __init__(self, n_heads: int, max_distance: int):
        super().__init__()
        self.n_heads = n_heads
        self.max_distance = max_distance
        self.bias = nn.Parameter(torch.zeros(n_heads, max_distance))

    def forward(self, T: int, device):
        idx = torch.arange(T, device=device)
        diff = idx.unsqueeze(0) - idx.unsqueeze(1)
        diff = diff.clamp(0, self.max_distance - 1)
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        b = self.bias[:, diff]
        b = b.masked_fill(causal.unsqueeze(0), 0.0)
        return b


def pos_family(cfg) -> str:
    """additive | rope | alibi | relative"""
    pe = cfg.positional_encoding_type
    if pe in ("rope",):
        return "rope"
    if pe in ("alibi",):
        return "alibi"
    if pe in ("relative_shaw",):
        return "relative"
    return "additive"


class ScaledDotProductCausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        fam = pos_family(cfg)
        if fam == "rope":
            assert self.head_dim % 2 == 0, "RoPE requires even head_dim"
            self.rotary = RotaryEmbedding(self.head_dim, cfg.max_position_embeddings)
        else:
            self.rotary = None
        if fam == "relative":
            self.rel_bias = RelativePositionBias(cfg.n_heads, cfg.relative_max_distance)
        else:
            self.rel_bias = None
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cfg = self.cfg
        fam = pos_family(cfg)

        if fam == "rope":
            q, k = self.rotary(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask(T, x.device), float("-inf"))
        if fam == "alibi":
            scores = scores + alibi_bias(self.n_heads, T, x.device, scores.dtype) * getattr(cfg, "alibi_slope_scale", 1.0)
        if fam == "relative":
            scores = scores + self.rel_bias(T, x.device).unsqueeze(0)
        w = self.dropout(F.softmax(scores, dim=-1))
        out = w @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class LocalWindowCausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.window = cfg.local_window_size
        fam = pos_family(cfg)
        if fam == "rope":
            assert self.head_dim % 2 == 0
            self.rotary = RotaryEmbedding(self.head_dim, cfg.max_position_embeddings)
        else:
            self.rotary = None
        if fam == "relative":
            self.rel_bias = RelativePositionBias(cfg.n_heads, cfg.relative_max_distance)
        else:
            self.rel_bias = None
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cfg = self.cfg
        fam = pos_family(cfg)
        if fam == "rope":
            q, k = self.rotary(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        i = torch.arange(T, device=x.device).unsqueeze(1)
        j = torch.arange(T, device=x.device).unsqueeze(0)
        dist = i - j
        invalid = (dist < 0) | (dist >= self.window)
        scores = scores.masked_fill(invalid, float("-inf"))
        if fam == "alibi":
            scores = scores + alibi_bias(self.n_heads, T, x.device, scores.dtype) * getattr(cfg, "alibi_slope_scale", 1.0)
        if fam == "relative":
            scores = scores + self.rel_bias(T, x.device).unsqueeze(0)
        w = self.dropout(F.softmax(scores, dim=-1))
        out = w @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class BlockSparseCausalSelfAttention(nn.Module):
    """Within each block of size Bk, full causal attention; no cross-block attention."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.block_size = cfg.sparse_block_size
        fam = pos_family(cfg)
        if fam == "rope":
            assert self.head_dim % 2 == 0
            self.rotary = RotaryEmbedding(self.head_dim, cfg.max_position_embeddings)
        else:
            self.rotary = None
        if fam == "relative":
            self.rel_bias = RelativePositionBias(cfg.n_heads, cfg.relative_max_distance)
        else:
            self.rel_bias = None
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        bk = self.block_size
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cfg = self.cfg
        fam = pos_family(cfg)
        if fam == "rope":
            q, k = self.rotary(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        cm = causal_mask(T, x.device)
        bi = torch.arange(T, device=x.device).unsqueeze(1) // bk
        bj = torch.arange(T, device=x.device).unsqueeze(0) // bk
        block_invalid = bi != bj
        invalid = cm | block_invalid
        scores = scores.masked_fill(invalid, float("-inf"))
        if fam == "alibi":
            scores = scores + alibi_bias(self.n_heads, T, x.device, scores.dtype) * getattr(cfg, "alibi_slope_scale", 1.0)
        if fam == "relative":
            scores = scores + self.rel_bias(T, x.device).unsqueeze(0)
        w = self.dropout(F.softmax(scores, dim=-1))
        out = w @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class LinearPerformerAttention(nn.Module):
    """Causal kernel linear attention phi(x)=elu(x)+1.

    Recurrence is O(T) per sequence with O(D^2) matmuls; **all batch and heads are
    vectorized**. (An earlier nested Python loop over batch×time caused multi-hour epochs.)
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        # q, k, v: (B, H, T, Dh). Causal linear attention via running KV and z.
        acc_kv = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=x.dtype)
        acc_k = torch.zeros(B, H, Dh, device=x.device, dtype=x.dtype)
        steps = []
        for t in range(T):
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]
            acc_kv = acc_kv + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            acc_k = acc_k + kt
            qt = q[:, :, t, :]
            num = torch.matmul(acc_kv, qt.unsqueeze(-1)).squeeze(-1)
            den = (qt * acc_k).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            steps.append((num / den).unsqueeze(2))
        out = torch.cat(steps, dim=2).transpose(1, 2).contiguous().view(B, T, D)
        out = self.dropout(out)
        return self.out_proj(out)


class GQACausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.group_size = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        fam = pos_family(cfg)
        if fam == "rope":
            assert self.head_dim % 2 == 0
            self.rotary = RotaryEmbedding(self.head_dim, cfg.max_position_embeddings)
        else:
            self.rotary = None
        if fam == "relative":
            self.rel_bias = RelativePositionBias(cfg.n_heads, cfg.relative_max_distance)
        else:
            self.rel_bias = None
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        cfg = self.cfg
        fam = pos_family(cfg)
        if fam == "rope":
            q, k = self.rotary(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask(T, x.device), float("-inf"))
        if fam == "alibi":
            scores = scores + alibi_bias(self.n_heads, T, x.device, scores.dtype) * getattr(cfg, "alibi_slope_scale", 1.0)
        if fam == "relative":
            scores = scores + self.rel_bias(T, x.device).unsqueeze(0)
        w = self.dropout(F.softmax(scores, dim=-1))
        out = w @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class MQAttention(nn.Module):
    """Multi-query attention — single KV head (MQA paper)."""

    def __init__(self, cfg):
        super().__init__()
        mc = copy.deepcopy(cfg)
        mc.n_kv_heads = 1
        self.inner = GQACausalSelfAttention(mc)

    def forward(self, x):
        return self.inner(x)


class SoftmaxFreeReluAttention(nn.Module):
    def __init__(self, cfg, eps=1e-6):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.eps = eps
        fam = pos_family(cfg)
        if fam == "rope":
            assert self.head_dim % 2 == 0
            self.rotary = RotaryEmbedding(self.head_dim, cfg.max_position_embeddings)
        else:
            self.rotary = None
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if pos_family(self.cfg) == "rope":
            q, k = self.rotary(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask(T, x.device), 0.0)
        w = F.relu(scores).pow(2)
        w = self.dropout(w)
        w = w / (w.sum(dim=-1, keepdim=True) + self.eps)
        out = w @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


ATTENTION_REGISTRY = {
    "scaled_dot_product": ScaledDotProductCausalSelfAttention,
    "local_window": LocalWindowCausalSelfAttention,
    "sparse_block": BlockSparseCausalSelfAttention,
    "linear_performer": LinearPerformerAttention,
    "gqa": GQACausalSelfAttention,
    "mqa": MQAttention,
    "softmax_free_relu": SoftmaxFreeReluAttention,
}


class CausalConv1d(nn.Module):
    """Depthwise-separable friendly causal Conv1d over sequence."""

    def __init__(self, d_model: int, kernel_size: int, groups: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=groups)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x).transpose(1, 2)
        return x


class StandardTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = ATTENTION_REGISTRY[cfg.attention_type](cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ConvBeforeAttentionBlock(nn.Module):
    """Hybrid (assignment §4): causal Conv1d before attention."""

    def __init__(self, cfg):
        super().__init__()
        self.conv = CausalConv1d(cfg.d_model, cfg.conv_kernel_size, groups=getattr(cfg, "conv_groups", 1))
        self.ln0 = nn.LayerNorm(cfg.d_model)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = ATTENTION_REGISTRY[cfg.attention_type](cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.conv(self.ln0(x))
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DepthwiseConvOnlyBlock(nn.Module):
    """Replace attention with causal depthwise conv + pointwise (subset replacement idea)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        dw = CausalConv1d(cfg.d_model, cfg.conv_kernel_size, groups=cfg.d_model)
        pw = nn.Linear(cfg.d_model, cfg.d_model)
        self.local = nn.Sequential(dw, pw)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.local(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


def build_blocks(cfg):
    n = cfg.n_layers
    bt = cfg.block_type
    blocks = []
    if bt == "standard_transformer_block":
        blocks = [StandardTransformerBlock(cfg) for _ in range(n)]
    elif bt == "conv_before_attention":
        blocks = [ConvBeforeAttentionBlock(cfg) for _ in range(n)]
    elif bt == "interleaved_conv_attention":
        for i in range(n):
            blocks.append(StandardTransformerBlock(cfg) if i % 2 == 0 else ConvBeforeAttentionBlock(cfg))
    elif bt == "alternate_attention_dwconv":
        for i in range(n):
            blocks.append(StandardTransformerBlock(cfg) if i % 2 == 0 else DepthwiseConvOnlyBlock(cfg))
    else:
        raise ValueError(f"Unknown block_type {bt}")
    return nn.ModuleList(blocks)


POSITIONAL_ENCODING_REGISTRY = {
    "learned": LearnedPositionalEmbedding,
    "sinusoidal": SinusoidalPositionalEncoding,
    "none": lambda max_len, d_model: IdentityPositionalEncoding(),
    "rope": lambda max_len, d_model: IdentityPositionalEncoding(),
    "alibi": lambda max_len, d_model: IdentityPositionalEncoding(),
    "relative_shaw": lambda max_len, d_model: IdentityPositionalEncoding(),
}


class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        max_pos = cfg.max_position_embeddings
        pos_cls = POSITIONAL_ENCODING_REGISTRY[cfg.positional_encoding_type]
        self.pos_enc = pos_cls(max_pos, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = build_blocks(cfg)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        x = self.dropout(self.pos_enc(self.tok_emb(idx)))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
