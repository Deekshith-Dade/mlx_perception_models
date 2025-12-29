# Copyright (c) Meta Platforms, Inc. and affiliates.
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    old_context_len: int = 8192
    rope_scale_factor: int = 1
    low_freq_factor: int = 1
    high_freq_factor: int = 32

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024

def repeat_kv(x: mx.array, n_rep: int, dim: int) -> mx.array:
    return mx.repeat(x, n_rep, axis=dim)


def reshape_for_broadcast(freqs_cis: mx.array, x: mx.array, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return mx.reshape(freqs_cis, shape=shape)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def apply_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    seq_dim: int,
    freqs_cis: mx.array
) -> Tuple[mx.array, mx.array]:
    xq_ = mx.reshape(xq, (*xq.shape[:-1], -1, 1, 2)) # B S H D -> B S H D/2 1 2
    xk_ = mx.reshape(xk, (*xk.shape[:-1], -1, 1, 2)) # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).astype(mx.float32) # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = mx.sum((xq_ * freqs_cis), axis=5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)
    
class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(mx.ones((dim, )))

    def _norm(self, x: mx.array):
        return x * mx.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: mx.array):
        # x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: int
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            head_dim * n_kv_heads,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            head_dim * n_kv_heads,
            bias=False,
        )
        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )
    
    def __call__(
        self,
        x: mx.array,
        freq_cis: mx.array,
        tok_idx: Optional[mx.array] = None,
        mask: Optional[str] = None,
        attn_impl: str = "sdpa",
    ) -> mx.array:
        # B, S, D
        bsz, seq_len, dim = x.shape
        xq = self.wq(mx.reshape(x, x.shape)) # TODO: WHY?
        xk = self.wq(mx.reshape(x, x.shape))
        xv = self.wq(mx.reshape(x, x.shape))
        
        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # pluggable kv cache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)
        
        # xk = repeat_kv(xk, self.heads_per_group, dim=2)
        # xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(0, 2, 1, 3), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, mx.array))
            # is_causal = (mask == "causal") if isinstance(mask, str) else False
            output = mx.fast.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                mask=mask
            )
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )
        
        output = self.wo(mx.reshape(output, output_shape))

        return output
    
    def reset_parameters(self, init_std=None, factor=1.0):
        pass

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) 
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
    
    def forward(self, x: mx.array) -> mx.array:
        # B S D
        x1 = self.w1(x)
        x3 = self.w3(x)
        output = self.w2(nn.silu(x1) * x3)
        return output
    
    def reset_parameters(self, init_std=None, factor=1.0):
        pass

class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        pass