from __future__ import annotations

import torch
from torch import nn, Tensor
import math
import einops
from cs336_basics.nn_utils import softmax

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        #return x @ self.weight.T
        return torch.einsum("...i, oi -> ...o", x, self.weight)
    
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]

class RMSnorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        result = (x * self.weight) * rms
        return result.to(in_dtype)
    

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.einsum("...i, oi -> ...o", x, self.w1_weight)
        silu = torch.sigmoid(gate) * gate
        value = torch.einsum("...i, oi -> ...o", x, self.w3_weight)
        hidden = silu * value
        return torch.einsum("...i, oi -> ...o", hidden, self.w2_weight)
    
class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))

        pos = torch.arange(max_seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x = einops.rearrange(x, "... (d p) -> ... d p", p=2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return einops.rearrange(torch.stack((out1, out2), dim=-1), "... d p -> ... (d p)")
    
class Scaled_dot_product_attention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        attn_scores = torch.einsum("... i d, ... j d -> ... i j", q, k) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = softmax(attn_scores, dim=-1)
        return torch.einsum("... i j, ... j d -> ... i d", attn_weights, v)
    

class Causal_multihead_self_attention(nn.modules):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_weight = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.k_weight = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.v_weight = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.out_weight = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
    )