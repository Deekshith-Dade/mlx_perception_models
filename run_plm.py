from PIL import Image
import time
from apps.plm.generate import load_consolidated_model_and_tokenizer
from apps.plm.generate_reg import GeneratorArgs, LMGenerator
from core.args import dataclass_from_dict
from core.transforms.image_transform import get_image_transform

# ckpt = "facebook/Perception-LM-1B"
ckpt = "/Users/deekshith/Documents/Projects/multimodal_models/mlx_perception_models/plm_models/facebook/Perception-LM-1B/original"
# ckpt = "facebook/Perception-LM-3B"
# ckpt = "facebook/Perception-LM-8B" 
model, tokenizer, config = load_consolidated_model_and_tokenizer(ckpt)

def generate(
    media_path="",
    question="Describe the image in details.",
    media_type="image",
    number_of_frames=4,
    number_of_tiles=1,
    temperature=0.0,
    top_p=None,
    top_k=None,
):
    prompts = []
    if media_type == "image":
        transform = get_image_transform(
            vision_input_type=(
                "vanilla" if number_of_tiles == 1 else config.data.vision_input_type
            ),
            image_res=model.vision_model.image_size,
            max_num_tiles=number_of_tiles,
        )
        image = Image.open(media_path).convert("RGB")
        image, _ = transform(image)
        prompts = ("Describe clearly, what you are seeing?", image)
    # elif media_type == "video":
    #     transform = get_video_transform(
    #         image_res=model.vision_model.image_size,
    #     )
    #     video_info = (media_path, number_of_frames, None, None, None)
    #     frames, _ = transform(video_info)
    #     prompts.append((question, frames))
    else:
        raise NotImplementedError(
            f"The provided generate function only supports image and video."
        )
        # Create generator
    gen_cfg = dataclass_from_dict(
        GeneratorArgs,
        {"temperature": temperature, "top_p": top_p, "top_k": top_k},
        strict=False,
    )
    generator = LMGenerator(gen_cfg, model, tokenizer=tokenizer)
    start_time = time.time()
    generation = generator.generate(prompts)
    end_time = time.time()

    total_tokens = len(tokenizer.encode(generation, False, False))
    tokens_per_second = total_tokens / (end_time - start_time)

    print("==============================================")
    print(f"\nPrompt {1}: {prompts[0]}")
    print(f"Generated Text: {generation}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print("==============================================")

media_path = "br.jpg"
question = "Describe the image in details."

print("Generating with 4 tiles + 1 tumb...")
generate(media_path=media_path, question=question, number_of_tiles=4, media_type="image")