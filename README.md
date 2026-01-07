# MLX Perception Models

This repository contains **MLX ports** of Meta's **Perception Encoder (PE-Core)** and **Perception Language Model (Perception-LM)** models, optimized for fast inference on Apple Silicon. The models are converted from PyTorch to MLX format and are hosted on the [mlx-community](https://huggingface.co/mlx-community) on Hugging Face.

> **📚 Original Repository:** For comprehensive documentation, benchmarks, training details, and the full model family (PE-Lang, PE-Spatial, PE-Audio-Visual), visit the original [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models) repository.

---

## ✨ Features

- 🍎 **Native Apple Silicon Support** — Optimized for M1/M2/M3/M4 chips via MLX
- 🚀 **Fast Inference** — Leverages unified memory architecture for efficient processing
- 📦 **Easy to Use** — Models auto-download from Hugging Face Hub
- 🔄 **Full Compatibility** — Same API and outputs as the original PyTorch models

---

## 📦 Available Models

### Perception Encoder (PE-Core) Models

All PE-Core models are available on [mlx-community](https://huggingface.co/mlx-community) and are automatically downloaded when you load them:

| Model | Image Size | Patch Size | Vision Layers | Vision Width | Params | Use Case |
|-------|------------|------------|---------------|--------------|--------|----------|
| `PE-Core-T16-384` | 384×384 | 16 | 12 | 192 | Tiny | Mobile / Edge |
| `PE-Core-S16-384` | 384×384 | 16 | 12 | 384 | Small | Fast inference |
| `PE-Core-B16-224` | 224×224 | 16 | 12 | 768 | Base | Balanced |
| `PE-Core-L14-336` | 336×336 | 14 | 24 | 1024 | Large | High quality |
| `PE-Core-G14-448` | 448×448 | 14 | 50 | 1536 | Giant | Best quality |

### Perception Language Model (Perception-LM) Models

All Perception-LM models are available on [mlx-community](https://huggingface.co/mlx-community) and are automatically downloaded when you load them:

| Model | Params | Use Case |
|-------|--------|----------|
| `mlx-community/Perception-LM-1B` | 1B | Fast inference, mobile |
| `mlx-community/Perception-LM-3B` | 3B | Balanced performance |
| `mlx-community/Perception-LM-8B` | 8B | High quality, detailed understanding |

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mlx_perception_models.git
cd mlx_perception_models

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Requirements
- Python ≥ 3.13
- macOS with Apple Silicon (M1/M2/M3/M4)
- MLX ≥ 0.30.1

---

## 🚀 Quick Start

Here's a simple example showing zero-shot image classification:

```python
import mlx.core as mx
from PIL import Image

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# List available models
print("Available models:", pe.CLIP.available_configs())

# Load model (auto-downloads from mlx-community on first use)
model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)

# Get preprocessing transforms
preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

# Load and preprocess image
image = preprocess(Image.open("docs/assets/br.jpg"))[None]

# Tokenize text prompts
text = tokenizer(["a diagram", "blade runner", "a dystopian city"])

# Get embeddings
image_features, text_features, logit_scale = model(image, text)

# Calculate similarity probabilities
text_probs = mx.softmax(logit_scale * image_features @ text_features.T, axis=-1)
print("Text probs:", text_probs)
# Output: Text probs: array([[0.00123, 0.987, 0.0118]], dtype=float32)
```

---

## 📖 About Perception Encoder

Perception Encoder (PE) is Meta's state-of-the-art family of vision encoders developed by Facebook Research. The PE-Core models are CLIP-style encoders that excel at:

- **Zero-shot Image Classification** — Classify images without task-specific training
- **Image-Text Retrieval** — Find images matching text queries and vice versa
- **Video Understanding** — Strong performance on video classification benchmarks
- **Foundation for VLMs** — Powers the Perception Language Model (PLM)

---

## 🤖 Running Perception-LM Models

**Perception-LM models are now working!** These vision-language models combine the PE-Core vision encoder with a language model to enable detailed visual understanding and image-to-text generation.

To run Perception-LM models, use the `run_plm.py` script. The models are available on [mlx-community](https://huggingface.co/mlx-community) and will be automatically downloaded when you run the script.

```bash
python run_plm.py
```

Alternatively, you can explore Perception-LM models interactively using the Jupyter notebook demo at `apps/plm/notebook_demos/image_grounding.ipynb`. This notebook provides examples of image grounding and visual question answering with Perception-LM models.

You can modify `run_plm.py` to:
- Change the model checkpoint (e.g., `mlx-community/Perception-LM-3B` or `mlx-community/Perception-LM-8B`)
- Specify different images and questions
- Adjust generation parameters (temperature, top_p, top_k)
- Configure the number of tiles for high-resolution image processing

The script supports:
- **Image Understanding** — Detailed image descriptions and visual question answering
- **High-Resolution Processing** — Multi-tile processing for large images
- **Fast Inference** — Optimized for Apple Silicon via MLX

---

## 🔧 Converting Models Locally

If you want to convert models from the original PyTorch checkpoints to MLX format locally, you can use the included `convert.py` script.

### Converting PE-Core Models

```python
from convert import pe_save_model_as_safetensors

# Convert a single model
pe_save_model_as_safetensors("PE-Core-L14-336", save_path="pe_models")

# Or convert all available models
model_names = [
    "PE-Core-T16-384",
    "PE-Core-S16-384", 
    "PE-Core-B16-224",
    "PE-Core-L14-336",
    "PE-Core-G14-448"
]

for model_name in model_names:
    pe_save_model_as_safetensors(model_name, save_path="pe_models")
```

This will:
1. Download the original PyTorch checkpoint from Hugging Face
2. Convert weights to MLX safetensors format
3. Save the model config as JSON
4. Create an index file for the weights

The converted models will be saved to the specified `save_path` directory with the following structure:

```
pe_models/
└── PE-Core-L14-336/
    ├── config.json
    ├── model.safetensors
    └── model.safetensors.index.json
```

### Converting Perception-LM Models

```python
from convert import plm_save_model_as_safetensors

# Convert a single model
plm_save_model_as_safetensors("facebook/Perception-LM-3B", save_path="plm_models")

# Or convert all available models
model_names = [
    "facebook/Perception-LM-1B",
    "facebook/Perception-LM-3B",
    "facebook/Perception-LM-8B"
]

for model_name in model_names:
    plm_save_model_as_safetensors(model_name, save_path="plm_models")
```

This will:
1. Download the original PyTorch checkpoint from Hugging Face Hub
2. Convert weights to MLX safetensors format
3. Save the model config (`params.json`) and tokenizer (`tokenizer.model`)
4. Create an index file for the weights

The converted models will be saved to the specified `save_path` directory with the following structure:

```
plm_models/
└── facebook/
    └── Perception-LM-3B/
        └── original/
            ├── consolidated.safetensors
            ├── model.safetensors.index.json
            ├── params.json
            └── tokenizer.model
```

---

## 🔗 Resources

- **Original Repository:** [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models)
- **MLX Community Models:** [huggingface.co/mlx-community](https://huggingface.co/mlx-community)
- **MLX Framework:** [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **Paper (PE):** [Perception Encoder: The best visual embeddings are not at the output of the network](https://arxiv.org/abs/2504.13181)
- **Paper (PLM):** [PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding](https://arxiv.org/abs/2504.13180)

---

## 📜 Citation

If you use these models in your research, please cite the original work:

```bibtex
@article{bolya2025PerceptionEncoder,
  title={Perception Encoder: The best visual embeddings are not at the output of the network},
  author={Daniel Bolya and Po-Yao Huang and Peize Sun and Jang Hyun Cho and Andrea Madotto and Chen Wei and Tengyu Ma and Jiale Zhi and Jathushan Rajasegaran and Hanoona Rasheed and Junke Wang and Marco Monteiro and Hu Xu and Shiyu Dong and Nikhila Ravi and Daniel Li and Piotr Doll{\'a}r and Christoph Feichtenhofer},
  journal={arXiv:2504.13181},
  year={2025}
}
```

---

## 📄 License

The PE-Core models are released under the Apache-2.0 license. See the [original repository](https://github.com/facebookresearch/perception_models) for full license details.

