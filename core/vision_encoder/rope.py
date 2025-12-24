from math import pi
from typing import Literal, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.core import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcast

def rotate_half(x):
    full_shape = x.shape
    d = full_shape[-1]
    x = x.reshape(*full_shape[:-1], d//2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    x = mx.stack((-x2, x1), axis=-1)
    x = x.reshape(*full_shape)
    return x

def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype
    
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]
    
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    
    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * mx.cos(freqs) * scale) + (rotate_half(t) * mx.sin(freqs) * scale)
    out = mx.concat([t_left, t, t_right], axis=-1)
    
    return out.astype(dtype)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        custom_freqs: Optional[mx.array] = None,
        freqs_for: Union[
            Literal["lang"], Literal["pixel"], Literal["constant"]
        ] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True
    ):
        super().__init__()

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32) / dim)
            )
        elif freqs_for == "pixel":
            freqs = mx.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = mx.ones(num_freqs).astype(mx.float32)
        
        self.cache_if_possible = cache_if_possible

        self._cached_freqs = None
        self._cached_scales = None

        self.freqs = freqs if learned_freq else mx.stop_gradient(freqs)
        self.learned_freq = learned_freq

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor
        
        self.use_xpos = use_xpos
        if not use_xpos:
            self._scale = None
            return
        
        scale = (mx.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        
        self.scale_base = xpos_scale_base
        self._scale = scale

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)
    
    def get_seq_pos(self, seq_len, dtype, offset=0):
        return (
            mx.arange(seq_len).astype(dtype) + offset
        ) / self.interpolate_factor
    
    def __call__(self,  t: mx.array, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible
            and not self.learned_freq
            and exists(seq_len)
            and self.freqs_for != "pixel"
        )
        
        if (
            should_cache
            and exists(self._cached_freqs)
            and (offset + seq_len) <= self._cached_freqs.shape[0]
        ):
            return self._cached_freqs[offset: (offset + seq_len)]
        
        freqs = self.freqs
        
        freqs = einsum("..., f -> ... f", t.astype(self.freqs.dtype), freqs)
        # freqs = repeat(freqs, "... f -> ... (f r)", r=2)
        freqs = mx.repeat(freqs, repeats=2, axis=-1)
        
        if should_cache:
            self._cached_freqs = freqs
        
        return freqs

class Rope2D:
    def __init__(self, dim, use_cls_token=False):
        self.dim = dim
        self.use_cls_token = use_cls_token
        self.grid_size = None
        self.freq = None
    
    def init_arrays(self):
        self.rope = RotaryEmbedding(self.dim // 2)
    
    def update_grid(self, grid_h, grid_w):
        if self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)
            
            if self.use_cls_token:
                grid_y_range = mx.arange(grid_h) + 1
                grid_x_range = mx.arange(grid_w) + 1
            else:
                grid_y_range = mx.arange(grid_h)
                grid_x_range = mx.arange(grid_w)

            freqs_y = mx.broadcast_to(self.rope(grid_y_range)[:, None], (grid_h, grid_w, self.dim // 2))
            freqs_x = mx.broadcast_to(self.rope(grid_x_range)[None, :], (grid_h, grid_w, self.dim // 2))
            freq = mx.concat([freqs_x, freqs_y], axis=-1).reshape(grid_h * grid_w, -1)

            if self.use_cls_token:
                freq = mx.concat(
                    [mx.zeros((1, freq.shape[-1])), freq], axis=0
                )
            self.freq = freq[None, ...]
    
    def __call__(self, q, k):
        q = apply_rotary_emb(self.freq[:, None, :, :], q)
        k = apply_rotary_emb(self.freq[:, None, :, :], k)
        
        return q, k
        