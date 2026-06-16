from __future__ import annotations

import torch
from torch import nn, Tensor
import math
from einops import rearrange, einsum
from jaxtyping import Float, Bool, Int
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
        return einsum(x, self.weight, "... i, o i -> ... o")
    
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

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids, :]

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
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: Tensor) -> Tensor:
        gate = einsum(x, self.w1.weight,"... i, o i -> ... o")
        silu = torch.sigmoid(gate) * gate
        value = einsum(x, self.w3.weight, "... i, o i -> ... o")
        hidden = silu * value
        return einsum(hidden, self.w2.weight, "... i, o i -> ... o")

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
        freqs = einsum(pos, inv_freq, "i, j -> i j")
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x = rearrange(x, "... (d p) -> ... d p", p=2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return rearrange(torch.stack((out1, out2), dim=-1), "... d p -> ... (d p)")
    
def Scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries, d_k"],
    K: Float[Tensor, " ... keys, d_k"],
    V: Float[Tensor, " ... keys, d_k"],
    mask: Bool[Tensor, " ... queries, keys"] | None = None
) -> Float[Tensor, " ... queries, d_k"]:
    
    d_k = K.shape[-1]
    atten_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        atten_scores = torch.where(mask, atten_scores, torch.tensor(float("-inf")))
    atten_weights = softmax(atten_scores, dim=-1)
    
    return einsum(atten_weights, V, "... queries keys, ... keys d_k -> ... queries d_k")
    

class Causal_multihead_self_attention(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        position_encoder: RoPE | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.position_encoder: RoPE | None = position_encoder
        
    def forward(self, x: Float[Tensor, " ... seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"] | None = None
        ) -> Float[Tensor, " ... seq_len d_v"]:
        
        *batch_dims, seq_len, d_model = x.size()
        assert d_model == self.d_model, f"Expected input feature dimension {self.d_model}, got {d_model}"
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q, K, V = (
            rearrange(X, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads) 
            for X in (Q, K, V)
        )
        
        if self.position_encoder is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            token_positions = rearrange(token_positions, "... seq_len -> ... 1 seq_len")
            
            Q = self.position_encoder(Q, token_positions)
            K = self.position_encoder(K, token_positions)
        
        causal_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).tril()
        causal_mask = causal_mask.__getitem__((None,) * len(batch_dims) + (... ,) )
        
        anten_output = Scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        anten_output = rearrange(anten_output, "... heads seq_len d_v -> ... seq_len (heads d_v)")
        
        return self.output_proj(anten_output)
    

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        position_encoder: RoPE | None = None,
    ):
        super().__init__()
        self.attention = Causal_multihead_self_attention(d_model, num_heads, position_encoder)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSnorm(d_model)
        self.ln2 = RMSnorm(d_model)
        
    def forward(self, x: torch.Tensor):
        
        x_attn = self.attention(self.ln1(x))
        x = x + x_attn
        x_ffn = self.ffn(self.ln2(x))
        x = x + x_ffn
        return x


class BasicsTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: None | float = 100000.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embedding = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.position_encoder = RoPE(theta, d_head, context_length) if theta is not None else None
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, position_encoder=self.position_encoder)
            for _ in range(num_layers)
        ])
        self.norm = RMSnorm(d_model)
        self.output_projection = Linear(d_model, vocab_size)
        
    def forward(self, x: Int[Tensor, " ... seq_len"]) -> Float[Tensor, " ... seq_len vocab_size"]:
        x = self.token_embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.output_projection(x)
    
    