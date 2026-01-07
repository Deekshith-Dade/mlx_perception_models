from dataclasses import dataclass, field
from typing import List, Optional
import mlx.core as mx
import mlx.nn as nn

from core.transformer import Attention
from .generate import sample_tokens

class KVCache:
    def __init__(self, bsz, seqlen, n_heads, head_dim, dtype=mx.float32):
        shape = (bsz, seqlen, n_heads, head_dim)
        self.k_cache = mx.zeros(shape, dtype=dtype)
        self.v_cache = mx.zeros(shape, dtype=dtype)
        self.offset = 0

    def update(self, xk, xv, tok_idx):
        b, s, h, d = xk.shape
        self.k_cache[:, self.offset:self.offset + s] = xk
        self.v_cache[:, self.offset:self.offset + s] = xv
        self.offset += s
        return self.k_cache[:, :self.offset], self.v_cache[:, :self.offset]

@dataclass
class GeneratorArgs:
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_gen_len: int = 256
    max_tokens: int = 11264
    until: List[str] = field(default_factory=list)
    # compile_prefilling: bool = False
    # reduce_generation_overhead: bool = False
    # show_progress: bool = False
    # dtype: Optional[str] = "bf16"
    # device: Optional[str] = "cuda"

class LMGenerator:
    def __init__(
        self,
        cfg: GeneratorArgs,
        model: nn.Module,
        tokenizer: nn.Module
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k

        self.max_gen_len = cfg.max_gen_len
        self.max_tokens = cfg.max_tokens
        self.tok_idxs = None
        self.current_tok_id = None
        self.until = cfg.until
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1

    def prefill(self, tokens, images, image_patch_text_ids, num_image_chunks):

        self.setup_kv_cache()
        image_pos_index = self.prepare_media_inputs(tokens, image_patch_text_ids=image_patch_text_ids)
        self.tok_idxs = mx.arange(tokens.shape[-1])
        prefill_out = self.model(
            tokens,
            tok_idx=self.tok_idxs,
            mask="causal",
            images=images,
            image_pos_index=image_pos_index,
            num_chunks=num_image_chunks,
            attn_impl="sdpa"
        )
        return prefill_out
    
    def setup_kv_cache(self):
        for module in self.model.modules():
            if isinstance(module, Attention):
                if not hasattr(module, "kv_cache"):
                    module.kv_cache = KVCache(
                        1,
                        self.max_tokens,
                        module.n_kv_heads,
                        module.head_dim,
                    )
                module.kv_cache.offset = 0

    def generate_next_token(self, current_token):
        current_token = current_token
        out = self.model(
            current_token,
            tok_idx=self.current_tok_id,
            attn_impl="sdpa",
        )
        self.current_tok_id += 1
        return out

    def prepare_media_inputs(
        self, tokens, image_patch_text_ids  
    ):
        image_pos_index = mx.full(tokens.shape, -1, dtype=mx.int32)
        num_image_tokens = len(image_patch_text_ids[0])
        image_indices = (
            mx.arange(num_image_tokens, dtype=mx.int32)
        )
        image_pos_index[0, image_patch_text_ids] = image_indices

        return image_pos_index
    
    def generate(self, prompt):
        
        question, image = prompt
        num_image_chunks = [image.shape[0]]
        text_ids, image_pos = self.tokenizer._tokenize_for_generation(
            question, image
        )
        encoded_prompt = mx.array(text_ids)[None]
        image_patch_text_ids = [image_pos]
        current_images = image

        prompt_logits = self.prefill(
            tokens=encoded_prompt,
            images=current_images,
            image_patch_text_ids=image_patch_text_ids,
            num_image_chunks=num_image_chunks
        )
        all_tokens = sample_tokens(
                prompt_logits, self.temperature, self.top_p, self.top_k
            )
        generated_tokens = []
        start_token = all_tokens[:, -1][None]
        generated_tokens.append(start_token[0].item())
        self.current_tok_id = mx.array([prompt_logits.shape[1]])

        current_token = start_token
        for i in range(1, self.max_gen_len):
            next_logits = self.generate_next_token(current_token)
            next_token =  sample_tokens(
                next_logits, self.temperature, self.top_p, self.top_k
            )
            tok = next_token[0]
            generated_tokens.append(tok.item())
            current_end_str = self.tokenizer.decode(
                generated_tokens[-self.max_until_size :]
            )
            contains_end_str = any(
                [e in current_end_str for e in self.until]
            )
            is_done = (
                contains_end_str
                or tok == self.tokenizer.eot_id
                or tok == self.tokenizer.eos_id
            )

            if is_done:
                break

            current_token = next_token
        
        generation = self.tokenizer.decode(generated_tokens)
        generation = generation.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
        return generation