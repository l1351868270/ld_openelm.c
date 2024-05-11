import argparse
import struct
from typing import Union, Optional
from transformers import AutoModelForCausalLM
import torch
import numpy as np

def write_fp32(tensor, file):
    file.write(tensor.detach().numpy().astype("float32").tobytes())

def write_int8(tensor, file):
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

# https://huggingface.co/apple/OpenELM-270M/blob/main/modeling_openelm.py
def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by the divisor
    It can be seen at:
    https://github.com/tensorflow/models/blob/2cfc99eff5e5eb729c6793d2f3d03aa1c9be2b15/research/slim/nets/mobilenet/mobilenet.py#L62

    Args:
        v: input value
        divisor: default to 8
        min_value: minimum divisor value
    Returns:
        new_v: new divisible value
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        head_len = 0
        file.write(struct.pack("i", 20240426))
        head_len += 1
        file.write(struct.pack("i", config.ffn_dim_divisor))
        head_len += 1
        file.write(struct.pack(f"{len(config.ffn_multipliers)}f", *config.ffn_multipliers))
        head_len += len(config.ffn_multipliers)
        file.write(struct.pack("i", config.head_dim))
        head_len += 1
        file.write(struct.pack("i", config.max_context_length))
        head_len += 1
        file.write(struct.pack("i", config.model_dim))
        head_len += 1
        file.write(struct.pack("i", config.num_gqa_groups))
        head_len += 1
        file.write(struct.pack(f"{len(config.num_kv_heads)}i", *config.num_kv_heads))
        head_len += len(config.num_kv_heads)
        file.write(struct.pack(f"{len(config.num_query_heads)}i", *config.num_query_heads))
        head_len += len(config.num_query_heads)
        file.write(struct.pack("i", config.num_transformer_layers))
        head_len += 1
        file.write(struct.pack(f"{len(config.qkv_multipliers)}f", *config.qkv_multipliers))
        head_len += len(config.qkv_multipliers)
        file.write(struct.pack("i", config.rope_freq_constant))
        head_len += 1
        file.write(struct.pack("i", config.rope_max_length))
        head_len += 1
        file.write(struct.pack("i", config.vocab_size))
        head_len += 1
        max_qkv_proj_dim = 0
        for layer_idx in range(config.num_transformer_layers):
            head_dim = config.head_dim
            q_heads = config.num_query_heads[layer_idx]
            k_heads = config.num_kv_heads[layer_idx]
            v_heads = config.num_kv_heads[layer_idx]
            out_features = (q_heads + k_heads + v_heads) * head_dim
            if out_features > max_qkv_proj_dim:
                max_qkv_proj_dim = out_features
        file.write(struct.pack("i", max_qkv_proj_dim))
        head_len += 1
        max_intermediate_dim = 0
        for layer_idx in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[layer_idx]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            if intermediate_dim > max_intermediate_dim:
                max_intermediate_dim = intermediate_dim
        file.write(struct.pack("i", max_intermediate_dim))
        head_len += 1
        print(f"header length: {head_len}")

        sd = model.state_dict()

        # embedder 4
        ll = config.model_dim * config.vocab_size
        file.write(struct.pack("i", ll))
        write_fp32(sd["transformer.token_embeddings.weight"], file)
        # print(sd["transformer.token_embeddings.weight"].shape) # # [model_dim, vocab_size] [32000, 1280]
        
        ll = config.num_transformer_layers * config.model_dim
        # print(f"ll={ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn_norm.weight"], file)
            # print(sd[f"transformer.layers.{i}.attn_norm.weight"])

        ll = 0
        for layer_idx in range(config.num_transformer_layers):
            head_dim = config.head_dim
            q_heads = config.num_query_heads[layer_idx]
            k_heads = config.num_kv_heads[layer_idx]
            v_heads = config.num_kv_heads[layer_idx]
            out_features = (q_heads + k_heads + v_heads) * head_dim
            ll += out_features * config.model_dim
        # print(f"ll={ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn.qkv_proj.weight"], file)
            # print(sd[f"transformer.layers.{i}.attn.qkv_proj.weight"].shape)

        ll = config.num_transformer_layers * config.head_dim
        # print(f"ll={ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn.q_norm.weight"], file)
            # print(sd[f"transformer.layers.{i}.attn.q_norm.weight"].shape)

        ll = config.num_transformer_layers * config.head_dim
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn.k_norm.weight"], file)
            # print(sd[f"transformer.layers.{i}.attn.k_norm.weight"].shape)

        ll = 0
        for layer_idx in range(config.num_transformer_layers):
            q_heads = config.num_query_heads[layer_idx]
            ll += q_heads * config.head_dim * config.model_dim
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn.out_proj.weight"], file)
            # print(sd[f"transformer.layers.{i}.attn.out_proj.weight"].shape)

        ll = config.num_transformer_layers * config.model_dim
        # print(f"ll={ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.ffn_norm.weight"], file)
            # print(sd[f"transformer.layers.{i}.ffn_norm.weight"].shape)

        ll = 0
        for layer_idx in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[layer_idx]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            ll += 2 * intermediate_dim * config.model_dim
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.ffn.proj_1.weight"], file)
            # print(sd[f"transformer.layers.{i}.ffn.proj_1.weight"].shape)
        
        ll = 0
        for layer_idx in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[layer_idx]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            ll += intermediate_dim * config.model_dim
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.ffn.proj_2.weight"], file)
            # print(sd[f"transformer.layers.{i}.ffn.proj_2.weight"].shape)

        
        ll = config.model_dim
        file.write(struct.pack("i", ll))
        write_fp32(sd[f"transformer.norm.weight"], file)
        # print(sd[f"transformer.norm.weight"].shape)


def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr

def q80_write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        head_len = 0
        file.write(struct.pack("i", 20240426))
        head_len += 1
        file.write(struct.pack("i", config.ffn_dim_divisor))
        head_len += 1
        file.write(struct.pack(f"{len(config.ffn_multipliers)}f", *config.ffn_multipliers))
        head_len += len(config.ffn_multipliers)
        file.write(struct.pack("i", config.head_dim))
        head_len += 1
        file.write(struct.pack("i", config.max_context_length))
        head_len += 1
        file.write(struct.pack("i", config.model_dim))
        head_len += 1
        file.write(struct.pack("i", config.num_gqa_groups))
        head_len += 1
        file.write(struct.pack(f"{len(config.num_kv_heads)}i", *config.num_kv_heads))
        head_len += len(config.num_kv_heads)
        file.write(struct.pack(f"{len(config.num_query_heads)}i", *config.num_query_heads))
        head_len += len(config.num_query_heads)
        file.write(struct.pack("i", config.num_transformer_layers))
        head_len += 1
        file.write(struct.pack(f"{len(config.qkv_multipliers)}f", *config.qkv_multipliers))
        head_len += len(config.qkv_multipliers)
        file.write(struct.pack("i", config.rope_freq_constant))
        head_len += 1
        file.write(struct.pack("i", config.rope_max_length))
        head_len += 1
        file.write(struct.pack("i", config.vocab_size))
        head_len += 1
        max_qkv_proj_dim = 0
        for layer_idx in range(config.num_transformer_layers):
            head_dim = config.head_dim
            q_heads = config.num_query_heads[layer_idx]
            k_heads = config.num_kv_heads[layer_idx]
            v_heads = config.num_kv_heads[layer_idx]
            out_features = (q_heads + k_heads + v_heads) * head_dim
            if out_features > max_qkv_proj_dim:
                max_qkv_proj_dim = out_features
        file.write(struct.pack("i", max_qkv_proj_dim))
        head_len += 1
        max_intermediate_dim = 0
        for layer_idx in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[layer_idx]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            if intermediate_dim > max_intermediate_dim:
                max_intermediate_dim = intermediate_dim
        file.write(struct.pack("i", max_intermediate_dim))
        head_len += 1
        print(f"header length: {head_len}")

        sd = model.state_dict()

        # embedder 4
        ll = config.model_dim * config.vocab_size
        file.write(struct.pack("i", ll))
        print(f"ll:{ll}")
        write_fp32(sd["transformer.token_embeddings.weight"], file)
        # w = sd["transformer.token_embeddings.weight"] # [model_dim, vocab_size] [32000, 1280]
        # q, s, err = quantize_q80(w, config.model_dim)
        # ll = config.model_dim * config.vocab_size
        # ll_s = config.vocab_size
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # write_int8(q, file)
        # write_fp32(s, file)
        # print(w.shape)
        # print(q.shape)
        # print(s.shape)
        
        ll = config.num_transformer_layers * config.model_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn_norm.weight"], file)
        # ll = config.num_transformer_layers * config.model_dim
        # ll_s = config.num_transformer_layers
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn_norm.weight"], config.model_dim)
        #     write_int8(q, file)
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn_norm.weight"], config.model_dim)
        #     write_int8(s, file)

        ll = 0
        ll_s = 0
        for layer_idx in range(config.num_transformer_layers):
            head_dim = config.head_dim
            q_heads = config.num_query_heads[layer_idx]
            k_heads = config.num_kv_heads[layer_idx]
            v_heads = config.num_kv_heads[layer_idx]
            out_features = (q_heads + k_heads + v_heads) * head_dim
            ll += out_features * config.model_dim
            ll_s += out_features
        print(f"ll:{ll}")
        print(f"ll_s:{ll_s}")
        file.write(struct.pack("i", ll))
        file.write(struct.pack("i", ll_s))
        for i in range(config.num_transformer_layers):
            q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.qkv_proj.weight"], config.model_dim)
            write_int8(q, file)
            print(f"transformer.layers.{i}.attn.qkv_proj.weight quantized {tuple(sd[f'transformer.layers.{i}.attn.qkv_proj.weight'].shape)} to Q4_0 with max error {err}")
        for i in range(config.num_transformer_layers):
            q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.qkv_proj.weight"], config.model_dim)
            write_fp32(s, file)

        ll = config.num_transformer_layers * config.head_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn.q_norm.weight"], file)
        # ll = config.num_transformer_layers * config.head_dim
        # ll_s = config.num_transformer_layers
        # # print(f"ll={ll}")
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.q_norm.weight"], config.head_dim)
        #     write_int8(q, file)
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.q_norm.weight"], config.head_dim)
        #     write_int8(s, file)

        ll = config.num_transformer_layers * config.head_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn.k_norm.weight"], file)
        # ll = config.num_transformer_layers * config.head_dim
        # ll_s = config.num_transformer_layers
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.k_norm.weight"], config.head_dim)
        #     write_int8(q, file)
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.k_norm.weight"], config.head_dim)
        #     write_int8(s, file)

        ll = 0
        ll_s = 0
        for layer_idx in range(config.num_transformer_layers):
            q_heads = config.num_query_heads[layer_idx]
            ll +=  config.model_dim * (q_heads * config.head_dim)
            ll_s += config.model_dim
            
        print(f"ll:{ll}")
        print(f"ll_s:{ll_s}")
        file.write(struct.pack("i", ll))
        file.write(struct.pack("i", ll_s))
        for i in range(config.num_transformer_layers):
            # print(sd[f"transformer.layers.{i}.attn.out_proj.weight"].shape)
            q_heads = config.num_query_heads[i]
            q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.out_proj.weight"], q_heads * config.head_dim) # [1280, 768] [1280, 1024]
            # print(q.shape)
            write_int8(q, file)
            print(f"transformer.layers.{i}.attn.out_proj.weight quantized {tuple(sd[f'transformer.layers.{i}.attn.out_proj.weight'].shape)} to Q4_0 with max error {err}")
        for i in range(config.num_transformer_layers):
            q_heads = config.num_query_heads[i]
            q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.out_proj.weight"], q_heads * config.head_dim)
            # print(s.shape)
            write_fp32(s, file)

        ll = config.num_transformer_layers * config.model_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.ffn_norm.weight"], file)
        # ll = config.num_transformer_layers * config.model_dim
        # ll_s = config.num_transformer_layers * config.model_dim
        # # print(f"ll={ll}")
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.ffn_norm.weight"], config.model_dim)
        #     write_int8(q, file)
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.ffn_norm.weight"], config.model_dim)
        #     write_int8(s, file)

        ll = 0
        ll_s = 0
        for layer_idx in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[layer_idx]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            ll += (2 * intermediate_dim) * config.model_dim
            ll_s += 2 * intermediate_dim
        print(f"ll:{ll}")
        print(f"ll_s:{ll_s}")
        file.write(struct.pack("i", ll))
        file.write(struct.pack("i", ll_s))
        for i in range(config.num_transformer_layers):
            q, s, err = quantize_q80(sd[f"transformer.layers.{i}.ffn.proj_1.weight"], config.model_dim) # [1536, 1280] [2048, 1280]
            write_int8(q, file)
            print(f"transformer.layers.{i}.ffn.proj_1.weight quantized {tuple(sd[f'transformer.layers.{i}.ffn.proj_1.weight'].shape)} to Q8_0 with max error {err}")
        for i in range(config.num_transformer_layers):
            q, s, err = quantize_q80(sd[f"transformer.layers.{i}.ffn.proj_1.weight"], config.model_dim)
            write_fp32(s, file)
        
        ll = 0
        ll_s = 0
        for layer_idx in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[layer_idx]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            ll += config.model_dim * intermediate_dim
            ll_s += config.model_dim
        print(f"ll:{ll}")
        print(f"ll_s:{ll_s}")
        file.write(struct.pack("i", ll))
        file.write(struct.pack("i", ll_s))
        for i in range(config.num_transformer_layers):
            # print(sd[f"transformer.layers.{i}.ffn.proj_2.weight"].shape)
            ffn_multiplier = config.ffn_multipliers[i]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            q, s, err = quantize_q80(sd[f"transformer.layers.{i}.ffn.proj_2.weight"], intermediate_dim) # [1280, 768] [1280, 1024]
            # print(q.shape)
            write_int8(q, file)
            print(f"transformer.layers.{i}.ffn.proj_2.weight quantized {tuple(sd[f'transformer.layers.{i}.ffn.proj_2.weight'].shape)} to Q8_0 with max error {err}")
        for i in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[i]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            q, s, err = quantize_q80(sd[f"transformer.layers.{i}.ffn.proj_2.weight"], intermediate_dim)
            # print(s.shape)
            write_fp32(s, file)

        ll = config.model_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        write_fp32(sd[f"transformer.norm.weight"], file)
        # ll = config.model_dim
        # ll_s = 1
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # q, s, err = quantize_q80(sd[f"transformer.norm.weight"], config.model_dim)
        # write_int8(q, file)
        # write_int8(s, file)


def quantize_q40(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 15.0
    # scale into range [-15, 15]
    quant = w / scale[:,None] 
    # scale into range [0, 30]
    quant = quant + 15.0
    # round to nearest integer
    int4val_low = torch.round(quant).to(torch.uint8) # low 4 bit
    assert torch.all(int4val_low >= 0)
    assert torch.all(int4val_low <= 30)
    int4val_high = torch.round(quant).to(torch.uint8) << 4 # high 4 bit
    int8val = int4val_low[:, ::2] + int4val_high[:, 1::2]
    # dequantize by rescaling
    fp32val = ((int4val_low.float() - 15.0) * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    # print(int4val_low)
    # print(int4val_high)
    # print(int8val)
    # print(fp32valr)
    # print(w)
    maxerr = err.max().item()
    return int8val, scale, maxerr

def q40_write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        head_len = 0
        file.write(struct.pack("i", 20240426))
        head_len += 1
        file.write(struct.pack("i", config.ffn_dim_divisor))
        head_len += 1
        file.write(struct.pack(f"{len(config.ffn_multipliers)}f", *config.ffn_multipliers))
        head_len += len(config.ffn_multipliers)
        file.write(struct.pack("i", config.head_dim))
        head_len += 1
        file.write(struct.pack("i", config.max_context_length))
        head_len += 1
        file.write(struct.pack("i", config.model_dim))
        head_len += 1
        file.write(struct.pack("i", config.num_gqa_groups))
        head_len += 1
        file.write(struct.pack(f"{len(config.num_kv_heads)}i", *config.num_kv_heads))
        head_len += len(config.num_kv_heads)
        file.write(struct.pack(f"{len(config.num_query_heads)}i", *config.num_query_heads))
        head_len += len(config.num_query_heads)
        file.write(struct.pack("i", config.num_transformer_layers))
        head_len += 1
        file.write(struct.pack(f"{len(config.qkv_multipliers)}f", *config.qkv_multipliers))
        head_len += len(config.qkv_multipliers)
        file.write(struct.pack("i", config.rope_freq_constant))
        head_len += 1
        file.write(struct.pack("i", config.rope_max_length))
        head_len += 1
        file.write(struct.pack("i", config.vocab_size))
        head_len += 1
        max_qkv_proj_dim = 0
        for layer_idx in range(config.num_transformer_layers):
            head_dim = config.head_dim
            q_heads = config.num_query_heads[layer_idx]
            k_heads = config.num_kv_heads[layer_idx]
            v_heads = config.num_kv_heads[layer_idx]
            out_features = (q_heads + k_heads + v_heads) * head_dim
            if out_features > max_qkv_proj_dim:
                max_qkv_proj_dim = out_features
        file.write(struct.pack("i", max_qkv_proj_dim))
        head_len += 1
        max_intermediate_dim = 0
        for layer_idx in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[layer_idx]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            if intermediate_dim > max_intermediate_dim:
                max_intermediate_dim = intermediate_dim
        file.write(struct.pack("i", max_intermediate_dim))
        head_len += 1
        print(f"header length: {head_len}")

        sd = model.state_dict()

        # embedder 4
        ll = config.model_dim * config.vocab_size
        file.write(struct.pack("i", ll))
        print(f"ll:{ll}")
        write_fp32(sd["transformer.token_embeddings.weight"], file)
        # w = sd["transformer.token_embeddings.weight"] # [model_dim, vocab_size] [32000, 1280]
        # q, s, err = quantize_q80(w, config.model_dim)
        # ll = config.model_dim * config.vocab_size
        # ll_s = config.vocab_size
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # write_int8(q, file)
        # write_fp32(s, file)
        # print(w.shape)
        # print(q.shape)
        # print(s.shape)
        
        ll = config.num_transformer_layers * config.model_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn_norm.weight"], file)
        # ll = config.num_transformer_layers * config.model_dim
        # ll_s = config.num_transformer_layers
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn_norm.weight"], config.model_dim)
        #     write_int8(q, file)
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn_norm.weight"], config.model_dim)
        #     write_int8(s, file)

        ll = 0
        ll_s = 0
        for layer_idx in range(config.num_transformer_layers):
            head_dim = config.head_dim
            q_heads = config.num_query_heads[layer_idx]
            k_heads = config.num_kv_heads[layer_idx]
            v_heads = config.num_kv_heads[layer_idx]
            out_features = (q_heads + k_heads + v_heads) * head_dim
            ll += out_features * config.model_dim
            ll_s += out_features
        print(f"ll:{ll}")
        print(f"ll_s:{ll_s}")
        file.write(struct.pack("i", ll))
        file.write(struct.pack("i", ll_s))
        for i in range(config.num_transformer_layers):
            w = sd[f"transformer.layers.{i}.attn.qkv_proj.weight"]
            q, s, err = quantize_q40(sd[f"transformer.layers.{i}.attn.qkv_proj.weight"], config.model_dim)
            write_int8(q, file)
            # print(q)
            print(f"transformer.layers.{i}.attn.qkv_proj.weight quantized {tuple(w.shape)} to Q4_0 with max error {err}")
        for i in range(config.num_transformer_layers):
            q, s, err = quantize_q40(sd[f"transformer.layers.{i}.attn.qkv_proj.weight"], config.model_dim)
            write_fp32(s, file)

        ll = config.num_transformer_layers * config.head_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn.q_norm.weight"], file)
        # ll = config.num_transformer_layers * config.head_dim
        # ll_s = config.num_transformer_layers
        # # print(f"ll={ll}")
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.q_norm.weight"], config.head_dim)
        #     write_int8(q, file)
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.q_norm.weight"], config.head_dim)
        #     write_int8(s, file)

        ll = config.num_transformer_layers * config.head_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.attn.k_norm.weight"], file)
        # ll = config.num_transformer_layers * config.head_dim
        # ll_s = config.num_transformer_layers
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.k_norm.weight"], config.head_dim)
        #     write_int8(q, file)
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.attn.k_norm.weight"], config.head_dim)
        #     write_int8(s, file)

        ll = 0
        ll_s = 0
        for layer_idx in range(config.num_transformer_layers):
            q_heads = config.num_query_heads[layer_idx]
            ll +=  config.model_dim * (q_heads * config.head_dim)
            ll_s += config.model_dim
            
        print(f"ll:{ll}")
        print(f"ll_s:{ll_s}")
        file.write(struct.pack("i", ll))
        file.write(struct.pack("i", ll_s))
        for i in range(config.num_transformer_layers):
            q_heads = config.num_query_heads[i]
            q, s, err = quantize_q40(sd[f"transformer.layers.{i}.attn.out_proj.weight"], q_heads * config.head_dim) # [1280, 768] [1280, 1024]
            write_int8(q, file)
            print(f"transformer.layers.{i}.attn.qkv_proj.weight quantized {tuple(sd[f'transformer.layers.{i}.attn.out_proj.weight'].shape)} to Q4_0 with max error {err}")
        for i in range(config.num_transformer_layers):
            q_heads = config.num_query_heads[i]
            q, s, err = quantize_q40(sd[f"transformer.layers.{i}.attn.out_proj.weight"], q_heads * config.head_dim)
            # print(s.shape)
            write_fp32(s, file)

        ll = config.num_transformer_layers * config.model_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.ffn_norm.weight"], file)
        # ll = config.num_transformer_layers * config.model_dim
        # ll_s = config.num_transformer_layers * config.model_dim
        # # print(f"ll={ll}")
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.ffn_norm.weight"], config.model_dim)
        #     write_int8(q, file)
        # for i in range(config.num_transformer_layers):
        #     q, s, err = quantize_q80(sd[f"transformer.layers.{i}.ffn_norm.weight"], config.model_dim)
        #     write_int8(s, file)

        ll = 0
        ll_s = 0
        for layer_idx in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[layer_idx]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            ll += (2 * intermediate_dim) * config.model_dim
            ll_s += 2 * intermediate_dim
        print(f"ll:{ll}")
        print(f"ll_s:{ll_s}")
        file.write(struct.pack("i", ll))
        file.write(struct.pack("i", ll_s))
        for i in range(config.num_transformer_layers):
            q, s, err = quantize_q40(sd[f"transformer.layers.{i}.ffn.proj_1.weight"], config.model_dim) # [1536, 1280] [2048, 1280]
            write_int8(q, file)
            print(f"transformer.layers.{i}.ffn.proj_1.weight quantized {tuple(sd[f'transformer.layers.{i}.ffn.proj_1.weight'].shape)} to Q4_0 with max error {err}")
        for i in range(config.num_transformer_layers):
            q, s, err = quantize_q40(sd[f"transformer.layers.{i}.ffn.proj_1.weight"], config.model_dim)
            write_fp32(s, file)
        
        ll = 0
        ll_s = 0
        for layer_idx in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[layer_idx]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            ll += config.model_dim * intermediate_dim
            ll_s += config.model_dim
        print(f"ll:{ll}")
        print(f"ll_s:{ll_s}")
        file.write(struct.pack("i", ll))
        file.write(struct.pack("i", ll_s))
        for i in range(config.num_transformer_layers):
            # print(sd[f"transformer.layers.{i}.ffn.proj_2.weight"].shape)
            ffn_multiplier = config.ffn_multipliers[i]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            q, s, err = quantize_q40(sd[f"transformer.layers.{i}.ffn.proj_2.weight"], intermediate_dim) # [1280, 768] [1280, 1024]
            # print(q.shape)
            write_int8(q, file)
            print(f"transformer.layers.{i}.ffn.proj_2.weight quantized {tuple(sd[f'transformer.layers.{i}.ffn.proj_2.weight'].shape)} to Q4_0 with max error {err}")
        for i in range(config.num_transformer_layers):
            ffn_multiplier = config.ffn_multipliers[i]
            intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))
            q, s, err = quantize_q40(sd[f"transformer.layers.{i}.ffn.proj_2.weight"], intermediate_dim)
            # print(s.shape)
            write_fp32(s, file)

        ll = config.model_dim
        print(f"ll:{ll}")
        file.write(struct.pack("i", ll))
        write_fp32(sd[f"transformer.norm.weight"], file)
        # ll = config.model_dim
        # ll_s = 1
        # file.write(struct.pack("i", ll))
        # file.write(struct.pack("i", ll_s))
        # q, s, err = quantize_q80(sd[f"transformer.norm.weight"], config.model_dim)
        # write_int8(q, file)
        # write_int8(s, file)


if __name__ == "__main__":
    # python export_openelm.py --filepath="openelm_270M_q80.bin" --dtype="q80"
    # python export_openelm.py --filepath="openelm_270M_q40.bin" --dtype="q40"
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="openelm_270M_q80.bin")
    parser.add_argument("--dtype", choices=("fp32", "q80", "q40"), type=str, default="q80")
    args = parser.parse_args()
    model_type = "apple/OpenELM-270M-Instruct"
    print("loading weights from pretrained openelm: %s" % model_type)

    model = AutoModelForCausalLM.from_pretrained(model_type, trust_remote_code=True)
    # print(model)
    sd = model.state_dict()
    # print(sd.keys())
    config = model.config
    # print(config)
    dtype = args.dtype
    filepath = args.filepath
    print(f"dtype:{dtype} filepath:{filepath}")
    if (dtype == "fp32"):
        write_model(model, filepath)
    if (dtype == "q80"):
        q80_write_model(model, filepath)
    if (dtype == "q40"):
        q40_write_model(model, filepath)
    
