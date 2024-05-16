'''
gcc --shared -fPIC -o openelm_kv.so openelm_kv.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py

++++++++++++++++++++++++++++ openelm_kv_v1.c ++++++++++++++++++++++++++++
gcc --shared -fPIC -o openelm_kv.so openelm_kv_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:56.51s, tokens:256, achieved 4.53 tokens/s

gcc --shared -fPIC -o openelm_kv.so -g -O0 openelm_kv_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:55.93s, tokens:256, achieved 4.58 tokens/s

gcc --shared -fPIC -o openelm_kv.so -g -O1 openelm_kv_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:12.26s, tokens:256, achieved 20.89 tokens/s

gcc --shared -fPIC -o openelm_kv.so -g -O2 openelm_kv_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:12.40s, tokens:256, achieved 20.64 tokens/s

gcc --shared -fPIC -o openelm_kv.so -g -O3 openelm_kv_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:12.02s, tokens:256, achieved 21.30 tokens/s

gcc --shared -fPIC -o openelm_kv.so -g -Ofast openelm_kv_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:7.06s, tokens:256, achieved 36.24 tokens/s

++++++++++++++++++++++++++++ openelm_kv_v2.c ++++++++++++++++++++++++++++
gcc --shared -fPIC -o openelm_kv.so openelm_kv_v2.c -lm -fopenmp -mavx -mavx2
OMP_NUM_THREADS=8 python run_kv.py
total time is:23.33s, tokens:256, achieved 10.97 tokens/s

gcc --shared -fPIC -o openelm_kv.so -O3 openelm_kv_v2.c -lm -fopenmp -mavx -mavx2
OMP_NUM_THREADS=8 python run_kv.py
total time is:8.77s, tokens:256, achieved 29.18 tokens/s

gcc --shared -fPIC -o openelm_kv.so -Ofast openelm_kv_v2.c -lm -fopenmp -mavx -mavx2
OMP_NUM_THREADS=8 python run_kv.py
total time is:7.36s, tokens:256, achieved 34.76 tokens/s

++++++++++++++++++++++++++++ openelm_kv_avx512.c ++++++++++++++++++++++++++++
gcc --shared -fPIC -o openelm_kv.so openelm_kv_avx512.c -lm -fopenmp -mavx -mavx2 -mavx512f
OMP_NUM_THREADS=8 python run_kv.py
total time is:17.55s, tokens:256, achieved 14.58 tokens/s

gcc --shared -fPIC -o openelm_kv.so -O3 openelm_kv_avx512.c -lm -fopenmp -mavx -mavx2 -mavx512f
OMP_NUM_THREADS=8 python run_kv.py
total time is:7.88s, tokens:256, achieved 32.49 tokens/s

gcc --shared -fPIC -o openelm_kv.so -Ofast openelm_kv_avx512.c -lm -fopenmp -mavx -mavx2 -mavx512f
OMP_NUM_THREADS=8 python run_kv.py
total time is:7.34s, tokens:256, achieved 34.88 tokens/s

++++++++++++++++++++++++++++ openelm_kv_q80_v1.c ++++++++++++++++++++++++++++
gcc --shared -fPIC -o openelm_kv.so openelm_kv_q80_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:64.93s, tokens:256, achieved 3.94 tokens/s

gcc --shared -fPIC -o openelm_kv.so -g -O1 openelm_kv_q80_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:11.35s, tokens:256, achieved 22.55 tokens/s

gcc --shared -fPIC -o openelm_kv.so -g -O2 openelm_kv_q80_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:12.09s, tokens:256, achieved 21.17 tokens/s

gcc --shared -fPIC -o openelm_kv.so -g -O3 openelm_kv_q80_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:12.01s, tokens:256, achieved 21.32 tokens/s

gcc --shared -fPIC -o openelm_kv.so -g -Ofast openelm_kv_q80_v1.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:4.11s, tokens:256, achieved 62.33 tokens/s

++++++++++++++++++++++++++++ openelm_kv_q80_v2.c ++++++++++++++++++++++++++++
gcc --shared -fPIC -o openelm_kv.so openelm_kv_q80_v2.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:71.63s, tokens:256, achieved 3.57 tokens/s

gcc --shared -fPIC -o openelm_kv.so -O1 openelm_kv_q80_v2.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:9.46s, tokens:256, achieved 27.06 tokens/s

gcc --shared -fPIC -o openelm_kv.so -O2 openelm_kv_q80_v2.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:10.49s, tokens:256, achieved 24.40 tokens/s

gcc --shared -fPIC -o openelm_kv.so -O3 openelm_kv_q80_v2.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:4.48s, tokens:256, achieved 57.20 tokens/s

gcc --shared -fPIC -o openelm_kv.so -Ofast openelm_kv_q80_v2.c -lm -fopenmp
OMP_NUM_THREADS=8 python run_kv.py
total time is:3.90s, tokens:256, achieved 65.60 tokens/s

++++++++++++++++++++++++++++ openelm.cu ++++++++++++++++++++++++++++
nvcc --shared -Xcompiler "-fPIC" -o openelm_kv.so openelm.cu -lm
python run_kv.py
total time is:6.78s, tokens:256, achieved 37.75 tokens/s

++++++++++++++++++++++++++++ openelm_v1.cu ++++++++++++++++++++++++++++
nvcc --shared -Xcompiler "-fPIC" -o openelm_kv.so openelm_v1.cu -lm
python run_kv.py
total time is:3.73s, tokens:256, achieved 68.72 tokens/s

++++++++++++++++++++++++++++ openelm_v2.cu ++++++++++++++++++++++++++++
nvcc --shared -Xcompiler "-fPIC" -o openelm_kv.so openelm_v2.cu -lm
total time is:4.13s, tokens:256, achieved 61.94 tokens/s

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
    batch = 1
    seq_len = len(tokenized_prompt)
    tokenized_prompt_c = [tokenized_prompt[0]]
    print(tokenized_prompt_c)
    max_seq_len = 256
    pos = 0
    init(batch, max_seq_len)
    # next = openelm_forward(tokenized_prompt_c, batch, seq_len, pos)
    output = []
    begin = time.time()
    while (pos < max_seq_len):
        if pos < seq_len:
            tokenized_prompt_c = [tokenized_prompt[pos]]
        else:
            tokenized_prompt_c = next
        next = openelm_forward(tokenized_prompt_c, batch, 1, pos)
        print(f"pos:{pos} {tokenized_prompt_c}")
        output.append(tokenized_prompt_c[0])
        pos += 1
    end = time.time()
    
    output.append(next[0])
    output_text = tokenizer.decode(
        output,
        skip_special_tokens=True
    )

    print(output_text)
    print(f"total time is:{end - begin:.2f}s, tokens:{max_seq_len}, achieved {max_seq_len / (end - begin):.2f} tokens/s")