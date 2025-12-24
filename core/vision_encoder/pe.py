from dataclasses import asdict
from functools import partial
from typing import Callable, Literal, Optional, Union
from logging import getLogger

import numpy as np
import mlx.core as mx
import mlx.nn as nn


from core.vision_encoder.rope import Rope2D
from core.vision_encoder.config import PEConfig, PETextConfig, PE_VISION_CONFIG, PE_TEXT_CONFIG
from core.vision_encoder.model_misc import DropPath

logger = getLogger()

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.dim = dim
        self.init_values = init_values
    
    def __call__(self, x):
        return x * self.gamma
    
    def init_arrays(self):
        self.gamma = self.init_values * mx.ones((self.dim))

class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_width: int,
    ):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, mlp_width)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(mlp_width, embed_dim)
    
    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        
class MHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        
        self.in_proj_weight = mx.zeros((3 * embed_dim, embed_dim))
        self.in_proj_bias = mx.zeros((3 * embed_dim,))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.scale = self.head_dim ** -0.5
    
    def __call__(self, q: mx.array, k: mx.array, v: mx.array, attn_mask: Optional[mx.array] = None):
        B, L, D = q.shape
        _, S, _ = k.shape
        H = self.num_heads
        Hd = self.head_dim
        if q is k and k is v:
            qkv = q @ self.in_proj_weight.T + self.in_proj_bias
            q_proj, k_proj, v_proj = mx.split(qkv, 3, axis=-1)
        else:
            Wq, Wk, Wv = mx.split(self.in_proj_weight, 3, axis=0)
            bq, bk, bv = mx.split(self.in_proj_bias, 3, axis=0)
            
            q_proj = q @ Wq.T + bq
            k_proj = k @ Wk.T + bk
            v_proj = v @ Wv.T + bv
        
        q_proj = q_proj.reshape((B, L, H, Hd)).transpose(0, 2, 1, 3)
        k_proj = k_proj.reshape((B, S, H, Hd)).transpose(0, 2, 1, 3)
        v_proj = v_proj.reshape((B, S, H, Hd)).transpose(0, 2, 1, 3)
        
        out = mx.fast.scaled_dot_product_attention(
            q_proj, k_proj, v_proj, scale=self.scale, mask=attn_mask
        )
        
        out = out.transpose(0, 2, 1, 3).reshape((B, L, D))
        
        out = self.out_proj(out)
        return out

class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        assert (
            self.embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.probe = mx.random.normal((1, num_probe, self.embed_dim))
        # self.attn = nn.MultiHeadAttention(
        #     self.embed_dim, self.num_heads 
        # )
        self.attn = MHA(self.embed_dim, self.num_heads)

        self.layernorm = norm_layer(self.embed_dim)
        self.mlp_width = int(self.embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim=self.embed_dim, mlp_width=self.mlp_width)

    def __call__(self, x: mx.array):
        batch, _, _ = x.shape

        #TODO: repeat in original implementation
        q = mx.broadcast_to(self.probe, (batch, self.probe.shape[1], self.embed_dim))
        x = self.attn(q, x, x)
        x = x + self.mlp(self.layernorm(x))
        
        return x

class SelfAttention(nn.Module):
    r"""
    Implements sequence packed attention and RoPe
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: Optional[nn.Module] = None
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        
        self.in_proj_weight = mx.zeros((3 * embed_dim, embed_dim))
        self.in_proj_bias = mx.zeros((3 * embed_dim,))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rope = rope
        self.scale = self.head_dim ** -0.5
    
    def init_arrays(self):
        glorot_init = nn.init.glorot_uniform()
        const_init = nn.init.constant(0.0)
        self.in_proj_weight = glorot_init(self.in_proj_weight)
        self.in_proj_bias = const_init(self.in_proj_bias)
        self.out_proj.bias = const_init(self.out_proj.bias)
    
    def __call__(
        self,
        x,
        attn_mask = None,
    ):
        b, seq_len, embed_dim = x.shape
        # proj = mx.dot(x, self.in_proj_weight.T) + self.in_proj_bias
        proj = mx.addmm(self.in_proj_bias, x, self.in_proj_weight.T)

        proj = mx.contiguous(mx.unflatten(proj, -1, (3, embed_dim))[None].transpose(3, 1, 2, 0, 4).squeeze(-2))

        q, k, v = proj[0], proj[1], proj[2]

        q = q.reshape((b, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = k.reshape((b, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape((b, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        
        if self.rope is not None:
            q, k = self.rope(q, k)
        
        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale
        )
        
        # attn = rearrange(attn, "b h s d -> b s (h d)")
        attn = attn.transpose(0, 2, 1, 3).reshape((b, seq_len, embed_dim))
        
        return self.out_proj(attn)

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()

        if rope:
            self.attn = SelfAttention(d_model, n_head, rope=rope)
        else:
            # self.attn = nn.MultiHeadAttention(d_model, n_head)
            self.attn = MHA(d_model, n_head)
        
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = MLP(embed_dim=d_model, mlp_width=mlp_width)
        
    def _call_attn(
        self,
        q_x: mx.array,
        attn_mask: Optional[mx.array] = None,
    ):
        if attn_mask is not None:
            if not attn_mask.dtype == mx.bool_:
                attn_mask = attn_mask.astype(q_x.dtype)
        
        if isinstance(self.attn, SelfAttention):
            return self.attn(q_x, attn_mask=attn_mask)
        else:
            return self.attn(q_x, q_x, q_x, attn_mask=attn_mask)
        
    def __call__(
        self,
        x: mx.array,
        attn_mask: Optional[mx.array] = None,
    ):
        x = x + self.drop_path1(
            self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask))
        )
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        # self.grad_checkpoint = False ?

        self.resblocks = [
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_path=drop_path,
                rope=rope,
            )
            for _ in range(layers)
        ]

    def truncate(self, layer_idx: int):
        """ Delete layers so the last layer is the given layer index. """
        self.layers = ((self.layers + layer_idx) % self.layers) + 1
        self.resblocks = self.resblocks[: self.layers]
    
    def __call__(
        self,
        x: mx.array,
        attn_mask: Optional[mx.array] = None,
        layer_idx: int = -1,
    ):
        stop_idx = (self.layers + layer_idx) % self.layers

        for i, r in enumerate(self.resblocks):
            x = r(x, attn_mask=attn_mask)
            if i == stop_idx:
                break
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        use_ln_pre: bool = True,
        use_ln_post: bool = True,
        ls_init_value: float = None,
        drop_path: float = 0.0,
        image_size: int = 448,
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        use_cls_token: bool = False,
        output_dim: Optional[int] = 1280,
        attn_pooler_heads: int = 8,
        pool_type: Literal["attn", "tok", "avg", "none"] = "attn",
    ):
        super().__init__()
        assert pool_type in ["attn", "tok", "avg", "none"]
        self.pool_type = pool_type
        self.patch_size = patch_size
        
        self.output_dim = output_dim or width
        self.proj_dim = output_dim
        self.heads = heads
        self.width = width
        self.layers = layers
        
        self.use_abs_posemb = use_abs_posemb
        self.use_cls_token = use_cls_token
        self.use_rope2d = use_rope2d
        self.image_size = image_size
        
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.rope = (
            Rope2D(
                dim=width // heads,
                use_cls_token=use_cls_token,
            )
            if self.use_rope2d
            else None
        )

        self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(width) if use_ln_post else nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=self.rope,
        )

        if pool_type == "attn":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=attn_pooler_heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        
        self.init_arrays()
    
    def init_arrays(self):
        def init_submodule_tensors(module: Union[nn.Module | list]):
            if isinstance(module, list):
                for sub_module in module:
                    init_submodule_tensors(sub_module)
            elif isinstance(module, nn.Module):
                for name, child in module.children().items():
                    init_fn = getattr(child, "init_arrays", None)
                    if callable(init_fn):
                        logger.debug(f"Initializing arrays for submodule: {name}")
                        init_fn()
                    init_submodule_tensors(child)
                    
        init_submodule_tensors(self)
        self.rope.init_arrays()

        # class embeddings and positional embeddings
        init_scale = self.width**-0.5
        if self.use_cls_token:
            self.class_embedding = init_scale * mx.random.normal((self.width, ))
        
        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = init_scale * mx.random.normal(
                (int(self.use_cls_token) + self.posemb_grid_size**2, self.width)
            )

        if self.proj_dim is not None:
            self.proj = init_scale * mx.random.normal((self.width, self.proj_dim))
    
    def load_ckpt(self, ckpt_path: str, verbose: bool = True):
        _sd = mx.load(ckpt_path)
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]
        
        # for backwards compatibility
        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}
        if any(k.startswith("visual.") for k in _sd.keys()):
            _sd = {k.replace("visual.", ""): v for k, v in _sd.items() if "visual" in k}
        
        m, u = self.load_weights(_sd, strict=False)

        if verbose or (m or u):
            logger.info(f"Missing keys for loading vision encoder: {m}")
            logger.info(f"Unexpected keys for loading vision encoder: {u}")
            print(f"Missing keys for loading vision encoder: {m}")
            print(f"Unexpected keys for loading vision encoder: {u}")
        
    def truncate(self, layer_idx: int):
        """ Delete layers so the last layer is the given layer index. """
        self.transformer.truncate(layer_idx)
        self.layers = self.transformer.layers
        
    @classmethod
    def from_config(
        cls,
        name: str,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        # if name not in PE_V
        pass

    @classmethod
    def available_configs(cls):
        pass
        # return list()
    
    def _sample_abs_posembed(self, grid_h: int, grid_w: int):
        """Interpolates the absolute position embedding if necessary."""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]
        
        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]
        
        pos_embed = (
            mx.contiguous(pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .transpose(0, 3, 1, 2))
        )

        scale_h = grid_h / self.posemb_grid_size
        scale_w = grid_w / self.posemb_grid_size
        upsample_fn = nn.Upsample(
            scale_factor=(scale_h, scale_w),
            mode="linear",
            align_corners=False
        )
        pos_embed = upsample_fn(pos_embed)
        pos_embed = mx.contiguous(pos_embed.transpose(0, 2, 3, 1).reshape(-1, self.width))
        
        if self.use_cls_token:
            pos_embed = mx.concat([cls_token_embed, pos_embed], axis=0)
            
        return pos_embed[None, ...]
    
    def _pool(self, x: mx.array):
        if self.pool_type == "tok":
            return x[:, 0]
        elif self.pool_type == "avg":
            return x.mean(axis=1)
        elif self.pool_type == "attn":
            return self.attn_pool(x).squeeze(1)
        else:
            raise NotImplementedError
    
    def forward_features(
        self,
        x: mx.array,
        norm: bool = False,
        layer_idx: int = -1,
        strip_cls_token: bool = False,
    ):
        batch, h, w, _ = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        x = self.conv1(x)
        
        x = x.reshape(batch, -1, self.width)

        if self.use_cls_token:
            x = mx.concat(
                [mx.broadcast_to(self.class_embedding.reshape(1, 1, -1), (batch, 1, self.width)), x], 
                axis=1
            )
        
        if self.use_abs_posemb:
            x = x + self._sample_abs_posembed(grid_h, grid_w)
        
        if self.use_rope2d:
            self.rope.update_grid(grid_h, grid_w)
        
        x = self.ln_pre(x)
        x = self.transformer(x, layer_idx=layer_idx)
        
        if norm:
            x = self.ln_post(x)
        
        if strip_cls_token and self.use_cls_token:
            x = x[:, 1:, :]
        
        return x
    
    def __call__(self, x: mx.array, **kwargs):
        x = self.forward_features(x, norm=True, **kwargs)
        x = self._pool(x)
        
        if self.proj_dim is not None:
            x = mx.matmul(x, self.proj)
        
        return x

class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 72,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        output_dim: int = 1280,
        no_causal_mask: bool = False,
        pad_id: int = 0,
        pool_type: str = "argmax",
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        output_tokens: bool = False,
        use_ln_post: bool = True,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.pool_type = pool_type
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.layers = layers
        
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = mx.zeros((self.num_pos, width))

        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()
        
        if no_causal_mask:
            self._attn_mask = None
        else:
            self._attn_mask = self.build_causal_mask()
        
        if pool_type == "attn" or pool_type == "attn_eos":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        
        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = mx.zeros((width, output_dim))
        
    def build_causal_mask(self):
        mask = mx.full((self.num_pos, self.num_pos), float("-inf"))
        mask = mx.triu(mask, k=1)
        return mask
    
    def load_ckpt(self, ckpt_path: str, verbose: bool = True):
        _sd = mx.load(ckpt_path)
        if "state_dict" in _sd:
            _sd = _sd["state_dict"]
        elif "weights" in _sd:
            _sd = _sd["weights"]
        
        # for backwards compatibility
        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}
        
        m, u = self.load_weights(_sd, strict=False)
        
        if verbose or (m or u):
            logger.info(f"Missing keys for loading text encoder: {m}")
            logger.info(f"Unexpected keys for loading text encoder: {u}")
            print(f"Missing keys for loading text encoder: {m}")
            print(f"Unexpected keys for loading text encoder: {u}")
    
    def text_global_pool(
        self, x, text: Optional[mx.array] = None, pool_type: str = "argmax"
    ):
        if pool_type == "first":
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == "last":
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == "argmax":
            assert text is not None
            pooled, tokens = x[mx.arange(x.shape[0]), text.argmax(axis=1)], x
        else:
            pooled = tokens = x
        
        return pooled, tokens
    
    def __call__(self, text):
        seq_len = text.shape[1]
        
        x = self.token_embedding(text)
        attn_mask = self._attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]
        
        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        pooled, tokens = self.text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = mx.matmul(pooled, self.text_projection)
        
        if self.output_tokens:
            return pooled, tokens
        
        return pooled
    

class CLIP(TextTransformer):
    def __init__(
        self,
        vision_cfg: PEConfig,
        text_cfg: PETextConfig,
        init_logit_scale: float = np.log(1 / 0.07)
    ):
        super(CLIP, self).__init__(**asdict(text_cfg))
        self.visual = VisionTransformer(**asdict(vision_cfg))
        self.image_size = self.visual.image_size
        self.logit_scale = mx.ones(()) * init_logit_scale
    
    def encode_image(self, image, normalize: bool = False):
        x = self.visual(image)
        if normalize:
            x = x / (mx.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-12)
        return x
    
    def encode_video(self, video, normalize: bool = False): # b n c h w
        b, n, c, h, w = video.shape
        frms = video.reshape(b * n, c, h, w)
        frm_feats = self.encode_image(frms, normalize=normalize)
        video_feats = frm_feats.reshape(b, n, -1)
        video_feats = video_feats.mean(axis=1)
        return video_feats
    
    def encode_text(self, text, normalize: bool = False):
        x = super().__call__(text)
        breakpoint()
        if normalize:
            x = x / (mx.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-12)
        return x

    def sanitize_weights(self, state_dict: dict):
        for key, value in state_dict.items():
            if "conv1" in key:
                state_dict[key] = value.transpose(0, 2, 3, 1)
        return state_dict

    def __call__(
        self,
        image: Optional[mx.array] = None,
        text: Optional[mx.array] = None,
    ):
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )

        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )
        
        return image_features, text_features, self.logit_scale.exp()
    
    @classmethod
    def from_config(
        cls,
        name: str,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        if name not in PE_VISION_CONFIG or name not in PE_TEXT_CONFIG:
            raise ValueError(f"{name} not found in configs.")
        
        model = cls(PE_VISION_CONFIG[name], PE_TEXT_CONFIG[name])
        if pretrained:
            model.load_ckpt()
        
        return model
    
    @classmethod
    def available_configs(cls):
        return [k for k in PE_VISION_CONFIG if k in PE_TEXT_CONFIG]

