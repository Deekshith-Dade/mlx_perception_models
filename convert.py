import os
from dataclasses import asdict
import json

from huggingface_hub import snapshot_download
import torch
import mlx.core as mx

from pathlib import Path
from core.vision_encoder.config import fetch_pe_checkpoint
from core.vision_encoder.config import PE_VISION_CONFIG, PE_TEXT_CONFIG

def pe_save_model_as_safetensors(model_name: str, save_path: str = "pe_models"):

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

def plm_save_model_as_safetensors(model_name: str, save_path: str = "plm_models"):
    assert model_name.startswith("facebook/Perception-LM")
    try:
        print(f"Downloading {model_name} from Hugging Face Hub...")
        ckpt_path = snapshot_download(model_name)
        ckpt_path = os.path.join(ckpt_path, "original")
        print(f"Downloaded to: {ckpt_path}")
    except Exception as e:
        # Handle exceptions, such as model not found on HF Hub
        print(f"An error occurred while downloading {model_name}: {e}")
        return
    
    save_path = Path(save_path)
    save_path = save_path / model_name / "original"
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(ckpt_path) / "consolidated.pth"
    state_dict = torch.load(model_path, map_location="cpu")

    weights = {}
    for k, v in state_dict.items():
        weights[k] = mx.array(v.to(torch.float32).numpy())
    
    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Save Model Weights
    model_path = save_path / "consolidated.safetensors"
    mx.save_safetensors(str(model_path), weights)

    for weight_name in weights.keys():
        index_data["weight_map"][weight_name] = "consolidated.safetensors"
    
    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    # Save Model index file
    with open(save_path / f"model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=2)
        
    # Save params.json and tokenizer.model
    # Save params.json and tokenizer.model from ckpt_path to save_path
    params_json_src = Path(ckpt_path) / "params.json"
    tokenizer_model_src = Path(ckpt_path) / "tokenizer.model"
    params_json_dst = save_path / "params.json"
    tokenizer_model_dst = save_path / "tokenizer.model"

    if params_json_src.exists():
        with open(params_json_src, "rb") as src_f, open(params_json_dst, "wb") as dst_f:
            dst_f.write(src_f.read())
    else:
        print(f"Warning: {params_json_src} not found, skipping.")

    if tokenizer_model_src.exists():
        with open(tokenizer_model_src, "rb") as src_f, open(tokenizer_model_dst, "wb") as dst_f:
            dst_f.write(src_f.read())
    else:
        print(f"Warning: {tokenizer_model_src} not found, skipping.")

    

def main():
    # model_name = "PE-Core-L14-336"
    # model_names = ["PE-Core-T16-384", "PE-Core-S16-384", "PE-Core-B16-224", "PE-Core-L14-336", "PE-Core-G14-448"]
    model_names = ["facebook/Perception-LM-1B"]
    for model_name in model_names:
        if "Core" in model_name:
            pe_save_model_as_safetensors(model_name, save_path="pe_models")
        elif "LM" in model_name:
            plm_save_model_as_safetensors(model_name, save_path="plm_models")
    
    

if __name__ == "__main__":
    main()