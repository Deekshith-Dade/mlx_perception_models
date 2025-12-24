from dataclasses import asdict
import json

import torch
import mlx.core as mx

from pathlib import Path
from core.vision_encoder.config import fetch_pe_checkpoint
from core.vision_encoder.config import PE_VISION_CONFIG, PE_TEXT_CONFIG

def save_model_as_safetensors(model_name: str, save_path: str = "pe_models"):

    print(f"Converting and saving model {model_name}...")
    save_path = Path(save_path)
    save_path = save_path / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    path = fetch_pe_checkpoint(model_name)
    state_dict = torch.load(path, map_location="cpu")

    weights = {}
    for k, v in state_dict.items():
        weights[k] = mx.array(v.numpy())
    
    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Save Model Weights
    model_path = save_path / "model.safetensors"
    mx.save_safetensors(str(model_path), weights)

    for weight_name in weights.keys():
        index_data["weight_map"][weight_name] = "model.safetensors"
    
    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    # Save Model index file
    with open(save_path / f"model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=2)

    # Save Model Config
    vision_config = PE_VISION_CONFIG[model_name]
    text_config = PE_TEXT_CONFIG[model_name]
    
    config_data = {
        "vision_config": asdict(vision_config),
        "text_config": asdict(text_config),
    }
    
    with open(save_path / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Model {model_name} saved to {save_path}")
    

def main():
    # model_name = "PE-Core-L14-336"
    model_names = ["PE-Core-T16-384", "PE-Core-S16-384", "PE-Core-B16-224", "PE-Core-L14-336", "PE-Core-G14-448"]
    for model_name in model_names:
        save_model_as_safetensors(model_name)
    
    

if __name__ == "__main__":
    main()