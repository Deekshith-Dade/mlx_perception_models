from PIL import Image
import numpy as np
import mlx.core as mx
from core.vision_encoder.tokenizer import SimpleTokenizer


def get_image_transform(
    image_size: int,
    center_crop: bool = False,
    interpolation = Image.Resampling.BILINEAR
):
    def transform(img: Image.Image) -> mx.array:
        if center_crop:
            # TODO: Does center crop in original implementation
            img = img.resize((image_size, image_size), resample=interpolation)
        else:
            img = img.resize((image_size, image_size), resample=interpolation)
        
        img = img.convert("RGB")
        
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = mx.array(x)

        # x = mx.transpose(x, (2, 0, 1))  # HWC to CHW

        x = (x - 0.5) / 0.5
        
        return x
    
    return transform

        

def get_text_tokenizer(context_length: int):
    return SimpleTokenizer(context_length=context_length)