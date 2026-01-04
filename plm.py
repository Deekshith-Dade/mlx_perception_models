from apps.plm.generate import load_consolidated_model_and_tokenizer

ckpt = "facebook/Perception-LM-1B"
# ckpt = "facebook/Perception-LM-3B"
# ckpt = "facebook/Perception-LM-8B" 
model, tokenizer, config = load_consolidated_model_and_tokenizer(ckpt)