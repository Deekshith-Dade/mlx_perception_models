# Copyright (c) Meta Platforms, Inc. and affiliates.
from enum import Enum
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096

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

def cross_entropy(pred: mx.array, target, **kwargs):
    nn.losses.nll_loss(
        nn.log_softmax(pred.flatten(end_axis=-2).astype(mx.float32), axis=-1),
        target.flatten(end_axis=-1),
        **kwargs
    ) 

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
    xq_out = mx.sum((xq_ * freqs_cis), axis=5).flatten(3) # B S H D/2 2 2 -> B S H D/2 2 -> B S H D
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seqlen: int = 1024,
        scale_factor: int = 1,
        low_freq_factor: int = 1,
        high_freq_factor: int = 32,
        old_context_len: int = 8192,
    ):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        if scale_factor != 1:
            self.low_freq_wavelen = old_context_len / low_freq_factor
            self.high_freq_wavelen = old_context_len / high_freq_factor
            assert self.low_freq_wavelen >= self.high_freq_wavelen
    
    def reset_parameters(self):
        self._freqs_cis = self.precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def apply_scaling(self, freqs):
        if self.scale_factor == 1:
            return freqs
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < self.high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > self.low_freq_wavelen:
                new_freqs.append(freq / self.scale_factor)
            else:
                assert self.low_freq_wavelen != self.high_freq_wavelen
                assert self.low_freq_wavelen != self.high_freq_wavelen
                smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_freqs.append(
                    (1 - smooth) * freq / self.scale_factor + smooth * freq
                )
        return mx.array(new_freqs, dtype=freqs.dtype)

    def precompute_freqs_cis(
        self,
        dim: int,
        end: int,
        theta: float = 10000.0
    ):
        # freqs = (1.0 / (theta ** mx.arange(0, dim, 2)[:dim // 2].astype(mx.float32) / dim))
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32) / dim))
        freqs = self.apply_scaling(freqs)

        t = mx.arange(end)
        freqs = mx.outer(t, freqs).astype(mx.float32)

        cos, sin = mx.cos(freqs), mx.sin(freqs) # t, freqs -> ()

        return mx.reshape(mx.stack((cos, -sin, sin, cos), axis=-1), shape=(*freqs.shape, 2, 2))

    def __call__(
        self, seqlen: Optional[int] = None, tok_idx: Optional[mx.array] = None
    ):
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self._freqs_cis[tok_idx]
        elif seqlen is not None:
            return self._freqs_cis[0:seqlen]
    
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
        self.weight = mx.ones((dim,))

    def _norm(self, x: mx.array):
        return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array):
        # x = probe.log_stats(x, "resid")
        output = self._norm(x.astype(mx.float32))
        return (output * self.weight.astype(mx.float32)).astype(x.dtype)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

class TiedLinear(nn.Module):
    def __init__(self, tied_module: nn.Module) -> None:
        super().__init__()
        self.tied_module = tied_module
        if not hasattr(tied_module, "weight"):
            raise AttributeError(
                "Provided module does not have attribute 'weight'. Please check your tied_module."
            )

    def __call__(self, x: mx.array) -> mx.array:
        return mx.matmul(x, self.tied_module.weight.T)

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
        xk = self.wk(mx.reshape(x, x.shape))
        xv = self.wv(mx.reshape(x, x.shape))
        
        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # pluggable kv cache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)
        
        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(0, 2, 1, 3), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, mx.array))
            # is_causal = (mask == "causal") if isinstance(mask, str) else False
            output = mx.fast.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                scale=1.0 / math.sqrt(self.head_dim),
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
    
    def __call__(self, x: mx.array) -> mx.array:
        # B S D
        x1 = self.w1(x)
        x3 = self.w3(x)
        output = self.w2(nn.silu(x1) * x3)
        return output
    
    def reset_parameters(self, init_std=None, factor=1.0):
        pass

class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify atleast head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta
        )

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier
        )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
    
    def __call__(
        self,
        x: mx.array,
        freq_cis: mx.array,
        tok_idx: Optional[mx.array] = None,
        mask: Optional[str] = None,
        attn_impl: str = "sdpa",
    ):
        h = x + self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    def init_weights(self, init_std=None, factor=1.0):
        pass

class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seq_len = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
            scale_factor=args.rope_scale_factor,
            low_freq_factor=args.low_freq_factor,
            high_freq_factor=args.high_freq_factor,
            old_context_len=args.old_context_len
        )

        self.layers = []
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        
    def __call__(
        self,
        h,
        tok_idx: Optional[mx.array] = None,
        mask: Optional[str] = None,
        attn_impl: str = "sdpa"
    ):
        freq_cis = self.rope_embeddings(seqlen=self.max_seq_len, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        return h
    
    def reset_parameters(self):
        self.rope_embeddings.reset_paramters()
    
    def init_weights(self):
        self.reset_parameters()
        pass

