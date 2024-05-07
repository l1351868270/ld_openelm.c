import struct
from typing import Union, Optional
from transformers import AutoModelForCausalLM

def write_fp32(tensor, file):
    file.write(tensor.detach().numpy().astype("float32").tobytes())

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
        print(f"ll={ll}")
        file.write(struct.pack("i", ll))
        for i in range(config.num_transformer_layers):
            write_fp32(sd[f"transformer.layers.{i}.ffn_norm.weight"], file)
            print(sd[f"transformer.layers.{i}.ffn_norm.weight"].shape)

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


if __name__ == "__main__":
    model_type = "apple/OpenELM-270M"
    print("loading weights from pretrained openelm: %s" % model_type)

    model = AutoModelForCausalLM.from_pretrained(model_type, trust_remote_code=True)
    # print(model)
    sd = model.state_dict()
    # print(sd.keys())
    config = model.config
    # print(config)
    filename = "openelm_270M.bin"
    write_model(model, filename)
