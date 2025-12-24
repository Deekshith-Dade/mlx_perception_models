import mlx.core as mx
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from core.vision_encoder.config import fetch_pe_checkpoint

print("CLIP configs:", pe.CLIP.available_configs())

model_name = "PE-Core-L14-336"
path = fetch_pe_checkpoint(model_name)
model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
weights_path = f"pe_models/{model_name}.safetensors"
model_state_dict = mx.load(weights_path)
weights = model.sanitize_weights(model_state_dict)
# breakpoint()
model.load_weights(model_state_dict, strict=True)
# print("Missing keys:", m)
# print("Unexpected keys:", u)


preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

image = preprocess(Image.open("docs/assets/br.jpg"))[None]
text = tokenizer(["a diagram", "a dog", "a dystopian city"])

image_features, text_features, logit_scale = model(image, text)
breakpoint()
text_probs = mx.softmax(logit_scale * image_features @ text_features.T, axis=-1)
print("Text probs:", text_probs)
# breakpoint()