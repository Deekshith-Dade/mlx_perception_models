# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import logging
from ntpath import exists
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from core.checkpoint import load_consolidated_checkpoint

logging.basicConfig(level=logging.INFO)

import mlx.core as mx
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from PIL import Image
import mlx.nn as nn
from tqdm import tqdm

from apps.plm.tokenizer import PLMTokenizer, Tokenizer, build_tokenizer
from apps.plm.transformer import LMTransformer, LMTransformerArgs
from core.args import dataclass_from_dict
from core.transformer import (Attention, causal_mask)
from core.transforms.image_transform import get_image_transform

def categorical_sampling(logits):
    return mx.random.categorical(logits, axis=-1)

def sample_top_p(probs: mx.array, p: float) -> mx.array:
    probs_sort, probs_idx = mx.sort(probs, axis=-1)
    probs_sort = probs_sort[..., ::-1]
    probs_idx = probs_idx[..., ::-1]
    probs_sum = mx.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    next_token = categorical_sampling(probs_sort)
    next_token = mx.take_along_axis(probs_idx, next_token, axis=-1)
    return next_token

def sample_top_k(probs, k):
    top_k_value, _ = mx.topk(probs, k, axis=-1)
    min_value_top_k = top_k_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs = mx.divide(probs, mx.sum(probs, axis=-1, keepdims=True))
    return categorical_sampling(probs)

def sample_tokens(logits:mx.array, temp=0.0, top_p=None, top_k=None):
    shape = logits.shape
    logits = logits.flatten(end_axis=-2)
    if temp > 0.0:
        probs = mx.softmax(logits / temp, axis=-1)
        
        if top_p is not None:
            probs = sample_top_p(probs, top_p)
        if top_k is not None:
            probs = sample_top_k(probs, top_k)
        else:
            next_token = categorical_sampling(probs)
    else:
        next_token = mx.argmax(logits, axis=-1)
    return next_token.reshape(shape[:-1])

def pack_prompts(prompts: List[int]):
    res = []
    lengths = []
    for i, p in enumerate(prompts):
        p = mx.array(p, dtype=mx.int64)
        l = p.shape[0]
        res.append(p)
        lengths.append(l)
    lengths = mx.array(lengths, dtype=mx.int64)
    res = mx.concat(res)
    return res, lengths

def batch_prompts(prompts, max_elements, lengths=None):
    batches = []
    current_batch = []
    current_count = 0

    for i in range(len(prompts)):
        prt = prompts[i]
        prompt_size = len(prt) if lengths is None else lengths[i]
        if current_count + prompt_size <= max_elements:
            current_batch.append(prt)
            current_count += prompt_size
        else:
            if current_batch:  # Add the current batch to batches
                batches.append(current_batch)
            # Start a new batch with the current prompt
            current_batch = [prt]
            current_count = prompt_size

    # Add the last batch if it contains any prompts
    if current_batch:
        batches.append(current_batch)

    return batches



def load_consolidated_model_and_tokenizer(ckpt):
    if os.path.exists(ckpt):
        ckpt_path = ckpt
    else:
        try:
            print(f"Downloading {ckpt} from Hugging Face Hub...")
            ckpt_path = snapshot_download(ckpt)
            ckpt_path = os.path.join(ckpt_path, "original")
            print(f"Downloaded to: {ckpt_path}")
        except Exception as e:
            # Handle exceptions, such as model not found on HF Hub
            print(f"An error occurred while downloading {ckpt}: {e}")
            return
    
    # Load params from model config
    config = os.path.join(ckpt_path, "params.json")
    config = OmegaConf.load(config)

    # Build Tokenizer
    tokenizer = build_tokenizer(
        config.data.tokenizer_name,
        (
            config.data.tokenizer_path
            if os.path.exists(config.data.tokenizer_path)
            else os.path.join(ckpt_path, config.data.tokenizer_path)
        ),
        pooling_ratio=config.model.pooling_ratio,
        patch_size=config.model.vision_model.patch_size,
    )

    # Build model and load the consolidate checkpoints
    model_args = dataclass_from_dict(LMTransformerArgs, config.model, strict=False)
    model = LMTransformer(model_args)
    load_consolidated_checkpoint(model, ckpt_path)
    model.eval()

    return model, tokenizer, config