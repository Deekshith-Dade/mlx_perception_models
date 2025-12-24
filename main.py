import mlx.core as mx
from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from core.vision_encoder.config import fetch_pe_checkpoint

print("CLIP configs:", pe.CLIP.available_configs())

model_names = ["PE-Core-T16-384", "PE-Core-S16-384", "PE-Core-B16-224", "PE-Core-L14-336", "PE-Core-G14-448"]
model_name = model_names[3]

model = pe.CLIP.from_config(model_name, pretrained=True)

preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

image = preprocess(Image.open("docs/assets/br.jpg"))[None]
text = tokenizer(["a diagram", "blade runner", "a dystopian city"])

image_features, text_features, logit_scale = model(image, text)
text_probs = mx.softmax(logit_scale * image_features @ text_features.T, axis=-1)
print("Text probs:", text_probs)