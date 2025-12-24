from PIL import Image
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

print("CLIP configs:", pe.CLIP.available_configs())

model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)


preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)

image = preprocess(Image.open("docs/assets/br.jpg"))[None]
text = tokenizer(["a diagram", "a dog", "a man"])

image_features, text_features, logit_scale = model(image, text)
breakpoint()