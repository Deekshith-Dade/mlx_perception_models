
from dataclasses import dataclass
from typing import Optional


@dataclass
class PEConfig:
    """ Vision Tower Config. """
    patch_size: int
    width: int
    layers: int
    heads: int
    mlp_ratio: int
    output_dim: Optional[int]

    ls_init_value: float = None
    drop_path: float = 0.0

    image_size: int = 224,
    use_abs_posemb: bool = True
    use_cls_token: bool = False
    use_rope2d: bool = True

    pool_type: str = "attn"
    attn_pooler_heads: int = 8

    use_ln_pre: bool = True
    use_ln_post: bool = True

@dataclass
class PETextConfig:
    """ Text Tower Config. """
    context_length: int
    width: int
    heads: int
    layers: int

    output_dim: int

    mlp_ratio: float = 4.0
    vocab_size: int = 49408

PE_VISION_CONFIG = {}
PE_TEXT_CONFIG = {}

#########################################
#                PE CORE                #
#########################################

PE_VISION_CONFIG["PE-Core-G14-448"] = PEConfig(
    image_size=448,
    patch_size=14,
    width=1536,
    layers=50,
    heads=16,
    mlp_ratio=8960 / 1536,
    pool_type="attn",
    output_dim=1280,
    use_cls_token=False,
)
PE_TEXT_CONFIG["PE-Core-G14-448"] = PETextConfig(
    context_length=72,
    width=1280,
    heads=20,
    layers=24,
    output_dim=1280
)


PE_VISION_CONFIG["PE-Core-L14-336"] = PEConfig(
    image_size=336,
    patch_size=14,
    width=1024,
    layers=24,
    heads=16,
    mlp_ratio=4.0,
    pool_type="attn",
    output_dim=1024,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-L14-336"] = PETextConfig(
    context_length=32,
    width=1024,
    heads=16,
    layers=24,
    output_dim=1024
)


PE_VISION_CONFIG["PE-Core-B16-224"] = PEConfig(
    image_size=224,
    patch_size=16,
    width=768,
    layers=12,
    heads=12,
    mlp_ratio=4.0,
    pool_type="attn",
    output_dim=1024,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-B16-224"] = PE_TEXT_CONFIG["PE-Core-L14-336"]




PE_VISION_CONFIG["PE-Core-S16-384"] = PEConfig(
    image_size=384,
    patch_size=16,
    width=384,
    layers=12,
    heads=6,
    mlp_ratio=4.0,
    pool_type="attn",
    output_dim=512,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-S16-384"] = PETextConfig(
    context_length=32,
    width=512,
    heads=8,
    layers=12,
    output_dim=512
)



PE_VISION_CONFIG["PE-Core-T16-384"] = PEConfig(
    image_size=384,
    patch_size=16,
    width=192,
    layers=12,
    heads=3,
    mlp_ratio=4.0,
    pool_type="attn",
    output_dim=512,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-T16-384"] = PE_TEXT_CONFIG["PE-Core-S16-384"]