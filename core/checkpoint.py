from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

def get_consolidated_ckpt_path(ckpt_dir: Path, mp_rank: int = 0, mp_size: int = 1):
    if mp_size == 1:
        assert mp_rank == 0
        no_rank_path = ckpt_dir / "consolidated.safetensors"
        if no_rank_path.exists():
            return no_rank_path
    return ckpt_dir / f"consolidated.{mp_rank:02d}.safetensors"

def load_consolidated_checkpoint(
    model: nn.Module,
    consolidated_path: str,
    vision_model_path: Optional[str] = None
):
    ckpt_path = Path(consolidated_path)
    cp_file = get_consolidated_ckpt_path(ckpt_path, mp_rank=0, mp_size=1)
    if cp_file.exists():
        st_dict = mx.load(cp_file)
        if "model" in st_dict:
            st_dict = st_dict["model"]
    else:
        checkpoint_files = sorted(ckpt_path.glob("consolidated.*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No consolidated checkpoint file found in {ckpt_path}."
            )
        st_dict = {}
        for ckpt_file in checkpoint_files:
            part = mx.load(ckpt_file)
            # If the checkpoint part is wrapped with "model", unwrap it
            if "model" in part:
                part = part["model"]
            # Merge the state dicts (assumes the keys are all unique or will correctly overwrite)
            st_dict.update(part)
    
    model.vision_projector.init_arrays()
    model.vision_model.init_arrays()
    model.rope_embeddings.reset_parameters()

    if vision_model_path is not None:
        model.vision_model.load_ckpt(vision_model_path)
    
    st_dict = model.vision_projector.sanitize_weights(st_dict)
    st_dict = model.vision_model.sanitize_weights(st_dict)
    try:
        model.load_weights(st_dict, strict=False)
    except Exception as e:
        # print(f"Exception while loading weights for text encoder: {e}")
        print(f"Exception while loading weights for text encoder: {e}")
        raise ValueError("Error loading weights for text encoder") from e
    
    
    