import struct
from transformers import AutoTokenizer

if __name__ == "__main__":
    tokenizer = "meta-llama/Llama-2-7b-hf"
    hf_access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=hf_access_token, use_fast=False)
    tokenized_prompt = tokenizer.encode("I believe the meaning of life is")
    print(tokenized_prompt)
    for i in tokenized_prompt:
        print(tokenizer.decode(i))

    filename = "openelm_prompt.bin"
    with open(filename, "wb") as file:
        file.write(struct.pack("i", 2)) # batch
        file.write(struct.pack("i", len(tokenized_prompt))) # length
        file.write(struct.pack(f"{len(tokenized_prompt)}i", *tokenized_prompt)) # version
        file.write(struct.pack(f"{len(tokenized_prompt)}i", *tokenized_prompt)) # number of tokens
        # max_token_length = 0
        # print(tokenizer.decode(31554).encode('utf-8'))
        # for i in range(0, n):
        #     # print(f"{i}: {tokenizer.decode(i)}")
        #     b = tokenizer.decode(i).encode('utf-8')
        #     if len(b) > max_token_length:
        #         max_token_length = len(b)
        # file.write(struct.pack("i", max_token_length))

        # for i in range(0, n):
        #     # print(f"{i}: {tokenizer.decode(i)}")
        #     b = tokenizer.decode(i).encode('utf-8')
        #     score = tokenizer.sp_model.get_score(i)
        #     l = len(b)
        #     file.write(struct.pack("fI", score, l))
        #     file.write(b)
    # print(f"wrote {filename}")