'''
gcc --shared -fPIC -o openelm_kv.so openelm_kv.c -lm -fopenmp

'''
import time
from transformers import AutoTokenizer
from ctypes import CDLL
from ctypes import c_int, POINTER
openelmlib = CDLL("./openelm_kv.so")

def init(batch: int, max_seq_len: int):
    openelmlib.c_init(c_int(batch), c_int(max_seq_len))

def openelm_forward(token, batch, seq_len, pos)->list:
    openelmlib.c_openelm_forward.restype = POINTER(c_int * batch)
    sample = openelmlib.c_openelm_forward(c_int(batch), c_int(seq_len), (c_int * len(token))(*token), c_int(pos)) 
    res = []
    for i in sample.contents:
        res.append(int(i))
    return res

def generate(batch: int, data: list, steps: int):
    openelmlib.c_generate(c_int(batch), c_int(len(data)), (c_int * len(data))(*data), c_int(steps))

if __name__ == '__main__':
    tokenizer = "meta-llama/Llama-2-7b-hf"
    hf_access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=hf_access_token, use_fast=False)
    tokenized_prompt = tokenizer.encode("Once upon a time there was a man named John")
    print([tokenized_prompt, tokenized_prompt])
    batch = 2
    seq_len = len(tokenized_prompt)
    tokenized_prompt_c = [tokenized_prompt[0], tokenized_prompt[0]]
    print(tokenized_prompt_c)
    max_seq_len = seq_len + 256
    pos = 0
    init(batch, max_seq_len)
    # next = openelm_forward(tokenized_prompt_c, batch, seq_len, pos)
    output = []
    begin = time.time()
    while (pos < max_seq_len):
        if pos < seq_len:
            tokenized_prompt_c = [tokenized_prompt[pos], tokenized_prompt[pos]]
        else:
            tokenized_prompt_c = next
        next = openelm_forward(tokenized_prompt_c, batch, 1, pos)
        print(f"pos:{pos} {tokenized_prompt_c}")
        output.append(tokenized_prompt_c[0])
        pos += 1
    end = time.time()
    print(f"total time is: {end - begin}s, tokens: {max_seq_len} {max_seq_len / (end - begin)} tokens/s")
    output.append(next[0])
    output_text = tokenizer.decode(
        output,
        skip_special_tokens=True
    )

    print(output_text)