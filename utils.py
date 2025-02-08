import torch
import torch.nn as nn
import logging
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
from typing import Tuple

from tqdm import tqdm

from lora_modules.lora import LoRALayer
from lora_modules.bit.bitlora import BitLoraLayer
from lora_modules.apcs.aplora import APLoraLayer

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

def load_lora_adapters(model, adapter_path):
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        config = json.load(f)

    model = add_lora_layers(
        model,
        lora_type=config["lora_type"],
        target_modules=config["target_modules"],
        rank=config["rank"],
        alpha=config["alpha"]
    )
    adapter_weights = torch.load(os.path.join(adapter_path, "adapter_model.bin"), weights_only=True)
    model.load_state_dict(adapter_weights, strict=False, assign=True)
    return model

def convert_dtypes(state_dict):
    for key in state_dict:
        if 'beta' in key and isinstance(state_dict[key].dtype, str):
            state_dict[key] = state_dict[key].to(dtype=torch.bfloat16)
    return state_dict

def load_lora_model_and_tokenizer(model_name: str, adapter_path, torch_dtype = torch.bfloat16) -> AutoModelForCausalLM:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        local_files_only=True,
        torch_dtype=torch_dtype,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id
    )

    model = load_lora_adapters(model, adapter_path)
    model.eval()
    return model, tokenizer

def load_model_and_tokenizer(model_name: str, torch_dtype = torch.bfloat16) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info(f"Loading model from {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    return model, tokenizer

def add_lora_layers(
    model: nn.Module,
    lora_type: str,
    target_modules: set,
    rank: int = 32,
    alpha: float = 32,
    dtype = torch.bfloat16
) -> nn.Module:
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model if not parent_name else model.get_submodule(parent_name)
                if lora_type == "lora":
                    lora_layer = LoRALayer  (
                        base_layer=module,
                        rank=rank,
                        alpha=alpha,
                        dtype=dtype
                    )   
                elif lora_type == "aplinear":
                    lora_layer = APLoraLayer  (
                        base_layer=module,
                        rank=rank,
                        alpha=alpha,
                        dtype=dtype
                    )
                elif lora_type == "bitlinear":
                    lora_layer = BitLoraLayer  (
                        base_layer=module,
                        rank=rank,
                        alpha=alpha,
                        dtype=dtype
                    )   
                else:
                    raise ValueError(f"Unsupported LoRA type: {lora_type}")
                setattr(parent, child_name, lora_layer)
    return model

def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

def generate_completions(model, tokenizer, prompts, generation_config, batch_size=8):
    completions = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        tokenized_prompts = tokenizer(
            prompts[i:i + batch_size],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        generation_toks = model.generate(
            input_ids=tokenized_prompts.input_ids.to(model.device),
            attention_mask=tokenized_prompts.attention_mask.to(model.device),
            generation_config=generation_config,
        )

        generation_toks = generation_toks[:, tokenized_prompts.input_ids.shape[-1]:]

        for generation_idx, generation in enumerate(generation_toks):
            completions.append({
                'prompt': prompts[i + generation_idx],
                'response': tokenizer.decode(generation, skip_special_tokens=True).strip()
            })

    return completions

def pack_bits(tensor):
    flat = tensor.flatten()
    padding_length = (8 - (flat.numel() % 8)) % 8
    if padding_length:
        flat = torch.cat([flat, torch.zeros(padding_length, dtype=torch.bool, device=flat.device)])
    
    packed = flat.view(-1, 8)
    bytes_tensor = torch.zeros(packed.size(0), dtype=torch.uint8, device=packed.device)
    
    for i in range(8):
        bytes_tensor |= (packed[:, i].to(torch.uint8) << i)
    
    return bytes_tensor, padding_length

def unpack_bits(bytes_tensor, original_shape, padding_length):
    bits = torch.zeros(bytes_tensor.numel() * 8, dtype=torch.bool, device=bytes_tensor.device)
    for i in range(8):
        bits[i::8] = (bytes_tensor & (1 << i)).bool()
    
    if padding_length:
        bits = bits[:-padding_length]
    return bits.reshape(original_shape)

def compress_weights(adapter_weights):
    compressed = OrderedDict()
    
    for key, tensor in adapter_weights.items():
        if 'lora_A.weight' in key or 'lora_B.weight' in key:
            binary = (tensor >= 0).to(torch.bool)
            packed_data, padding = pack_bits(binary)
            compressed[key] = {
                'data': packed_data,
                'shape': tensor.shape,
                'padding': padding
            }
        elif 'scale' in key:
            compressed[key] = tensor
            
    return compressed

def decompress_weights(compressed_weights):
    decompressed = OrderedDict()
    
    for key, value in compressed_weights.items():
        if isinstance(value, dict):
            binary = unpack_bits(value['data'], value['shape'], value['padding'])
            decompressed[key] = binary.float() * 2 - 1
        else:
            decompressed[key] = value
            
    return decompressed
