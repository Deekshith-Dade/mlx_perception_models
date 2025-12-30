from dataclasses import dataclass, field
import itertools
import logging
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from core.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    TiedLinear,
    cross_entropy
)

from core.utils import InitArgs
from core.vision_encoder.pe import VisionTransformeras as PE_VisionTransformer
from core.vision_projector.mlp import MLPProjector

logger = logging.getLogger(__name__)

def create_causal_mask(seqlen, attn_impl, sliding_window):
    if attn_impl == "sdpa":
        return "causal"
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

@dataclass
class LMTransformerArgs(BaseTransformerArgs):

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False

    sliding_window: Optional[int] = None

    freeze_language_model: Optional[bool] = False
    freeze_vision_model: Optional[bool] = False

    vision_model: Optional[Dict[str, Any]] = None

    mlp_init: InitArgs = field(default_factory=InitArgs)
    pooling_ratio: int = 1
    remove_vision_class_token: bool = True

    attn_impl: str = "sdpa"

class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False
        )

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(
                args.dim,
                args.vocab_size,
                bias=False
            )
        
        if args.vision_model:
            logger.info(
                f"Initializing PE_VisionTransformer with args: {args.vision_model}"
            )
            self.vision_model = PE_VisionTransformer(**args.vision_model, output_dim=None)
            self.vision_projector = MLPProjector(args)
        
        self.freeze_vision_model = args.freeze_language_model
        self.freeze_language_model = args.freeze_language_model
        
    def train(self, mode: bool = True):
        super().train(mode=mode)
        for name, param in self.parameters():
            if "vision_model" in name:
                param.requires_grad = mode and not self.freeze_vision_model
            elif "vision_projector" in name:
                param.requires_grad = mode
            else:
                param.requires_grad = mode and not self.freeze_language_model
        return self
    
    def __call__(
        self,
        token_values: mx.array,
        target: Optional[mx.array] = None,
        tok_idx: Optional[mx.array] = None,
        mask: Optional[Union[mx.array, str]] = None,
        images: Optional[mx.array] = None,
        image_pos_index: Optional[mx.array] = None,
        loss_mask: Optional[mx.array] = None,
        aspect_ratios: Optional[mx.aray] = None,
        num_chunks: List[int] = [1],
        media_type: List[str] = ["multi_image"],
        attn_impl: str = "sdpa",
    ):
        _, seqlen = token_values.shape

        h = self.tok_embeddings(token_values) # B, seqlen, z

        if images is not None:
            h_img = self.vision_model(images, strip_cls_token=True)
            h_img = self.vision_projector(h_img)

            h = self.stitch_images_into_text(
                h,
                h_img,
                image_pos_index,
                num_chunks=num_chunks,
                media_type=media_type
            )
        
        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        h = super().__call__(h, tok_idx, mask=mask, attn_impl=attn_impl)

        logits = self.output(self.norm(h))

        if target is not None:
            logits = logits[loss_mask]
            target = target[loss_mask]
            return cross_entropy(logits, target)
        else:
            return logits
    
    def reset_parameters(self):
        return super().reset_parameters()   
        pass

    def stitch_images_into_text(
        self,
        h_tok: mx.array,
        h_img: List[mx.array],
        iamge_pos_index: mx.array,
        num_chunks: List[int],
        media_type: List[str]
    ):
        cumulative_indices=list(itertools.accumulate(num_chunks, initial=0))

        non_text_indices = [
            
        ]


