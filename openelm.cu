/*

nvcc -o openelm_cu openelm.cu -lm
./openelm_cu
seconds:7.500000s tokens:256 achieved tok/s: 34.133333

nvcc -o openelm_cu -O3 openelm.cu -lm
./openelm_cu
seconds:8.782000s tokens:256 achieved tok/s: 29.150535

python generate_openelm.py --device=cuda --max_length=256
Generation took 4.44 seconds.
 256 tokens.
 57.7 tokens/s.
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

extern "C" {
    void c_init(int batch, int max_seq_len);
    int* c_openelm_forward(int batch, int seq_len, int *data, int pos);
    void c_generate(int batch, int seq_len, int *data, int steps);
    void c_chat ();
}

typedef struct {
    int ffn_dim_divisor;
    float ffn_multipliers[16];
    int head_dim;
    int max_context_length;
    int model_dim;
    int num_gqa_groups;
    int num_kv_heads[16];
    int num_query_heads[16];
    int num_transformer_layers;
    float qkv_multipliers[2];
    int rope_freq_constant;
    int rope_max_length;
    int vocab_size;
    int max_qkv_proj_dim;
    int max_intermediate_dim;
} OpenELMConfig;

typedef struct {
    float *token_embeddings; // transformer.token_embeddings.weight
    float *attn_norm;        // transformer.layers.{i}.attn_norm.weight
    float *qkv_proj;         // transformer.layers.{i}.attn.qkv_proj.weight
    float *q_norm;           // transformer.layers.{i}.attn.q_norm.weight
    float *k_norm;           // transformer.layers.{i}.attn.k_norm.weight
    float *out_proj;         // transformer.layers.{i}.attn.out_proj.weight
    float *proj_1;           // transformer.layers.{i}.ffn.proj_1.weight
    float *proj_2;           // transformer.layers.{i}.ffn.proj_2.weight
    float *ffn_norm;         // transformer.layers.{i}.ffn_norm.weight
    float *norm;             // transformer.norm.weight
} OpenELMWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *x_qkv_proj;
    float *ihb;
    float *ihb2;
    float *hb;
    float *att;
    float *logits;
    int *next;
    int *token;
    int *next_cpu;

    float *q;
    float *key_cache;
    float *value_cache;

    int batch;
    int seq_len;
    int max_seq_len;

    int qkv_proj_offset;
    int out_proj_offset;
    int proj_1_offset;
    int proj_2_offset;

    int max_q_heads;
    int max_kv_heads;
} RunState;

typedef struct {
    OpenELMConfig config;
    RunState state;
    OpenELMWeights weights;
    OpenELMWeights weights_cu;
    int num_parameters;
    float *params_memory;
} OpenELM;

void malloc_run_state(RunState* s, OpenELMConfig* p) {
    int seq_len = s->max_seq_len;

    cudaMalloc((void**)&s->x, s->batch * p->model_dim * sizeof(float));
    cudaMalloc((void**)&s->xb2, s->batch * p->model_dim * sizeof(float));
    
    int q_heads = 0;
    int k_heads = 0;
    int v_heads = 0;
    for (int i = 0; i < p->num_transformer_layers; i++) {
        if (p->num_query_heads[i] > q_heads) {
            q_heads = p->num_query_heads[i];
        }
        if (p->num_kv_heads[i] > k_heads) {
            k_heads = p->num_kv_heads[i];
        }
        v_heads = k_heads;
    }
    s->max_q_heads = q_heads;
    s->max_kv_heads = k_heads;

    int max_xb_dim = p->model_dim;
    if (max_xb_dim < q_heads * p->head_dim) {
        max_xb_dim = q_heads * p->head_dim;
    }
    cudaMalloc((void**)&s->xb, s->batch * max_xb_dim * sizeof(float));
    cudaMalloc((void**)&s->att, s->batch * q_heads * seq_len * sizeof(float));
    cudaMalloc((void**)&s->ihb, s->batch * 2 * p->max_intermediate_dim * sizeof(float));
    cudaMalloc((void**)&s->ihb2, s->batch * p->max_intermediate_dim * sizeof(float));
    cudaMalloc((void**)&s->hb, s->batch * p->model_dim * sizeof(float));
    cudaMalloc((void**)&s->logits, s->batch * p->vocab_size * sizeof(float));
    cudaMalloc((void**)&s->next, s->batch * sizeof(int));
    cudaMalloc((void**)&s->token, s->batch * sizeof(int));
    s->next_cpu = (int*)malloc(s->batch * sizeof(int));
    cudaMalloc((void**)&s->q, s->batch * q_heads * p->head_dim * sizeof(float));
    cudaMalloc((void**)&s->key_cache, s->batch * p->num_transformer_layers * seq_len * k_heads * p->head_dim * sizeof(float));
    cudaMalloc((void**)&s->value_cache, s->batch * p->num_transformer_layers * seq_len * v_heads * p->head_dim * sizeof(float));
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
}

void memory_map_weights(OpenELMWeights *w, OpenELMConfig* p, float* ptr) {
    int ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("++++++++++++--------%d\n", ll);
    ptr += 1;
    w->token_embeddings = ptr;
    ptr += ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("++++++++++++--------%d\n", ll);
    ptr += 1;
    w->attn_norm = ptr;
    ptr += ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    ptr += 1;
    // printf("++++++++++++--------%d\n", ll);
    w->qkv_proj = ptr;
    ptr += ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    ptr += 1;
    w->q_norm = ptr;
    ptr += ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    ptr += 1;
    w->k_norm = ptr;
    ptr += ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    ptr += 1;
    w->out_proj = ptr;
    ptr += ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    ptr += 1;
    w->ffn_norm = ptr;
    ptr += ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    ptr += 1;
    w->proj_1 = ptr;
    ptr += ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    ptr += 1;
    w->proj_2 = ptr;
    ptr += ll;
    cudaMemcpy(&ll, ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("++++++++++++--------%d\n", ll);
    ptr += 1;
    w->norm = ptr;
}


void openelm_build_from_checkpoint(OpenELM *model, const char* checkpoint_path) {
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) {
        printf("Error opening model file\n");
    }

    size_t file_size = 0;
    fseek(model_file, 0, SEEK_END);
    file_size = ftell(model_file);
    fseek(model_file, 0, SEEK_SET);
    printf("file_size is: %ld\n", file_size);

    int model_magic;
    fread(&model_magic, sizeof(int), 1, model_file);
    if (model_magic != 20240426) {
        printf("Bad magic model file\n");
    }
    printf("model magic is: %d\n", model_magic);

    fread(&model->config, sizeof(int), sizeof(model->config) / sizeof(int), model_file);
    printf("config ffn_dim_divisor is: %d\n", model->config.ffn_dim_divisor);
    printf("config ffn_multipliers is: ");
    for (int i = 0; i < 16; i++) {
        printf("%f ", model->config.ffn_multipliers[i]);
    }
    printf("\n");
    printf("config head_dim is: %d\n", model->config.head_dim);
    printf("config model_dim is: %d\n", model->config.model_dim);
    printf("config num_gqa_groups is: %d\n", model->config.num_gqa_groups);
    printf("config num_kv_heads is: ");
    for (int i = 0; i < 16; i++) {
        printf("%d ", model->config.num_kv_heads[i]);
    }
    printf("\n");
    printf("config num_query_heads is: ");
    for (int i = 0; i < 16; i++) {
        printf("%d ", model->config.num_query_heads[i]);
    }
    printf("\n");
    printf("config num_transformer_layers is: %d\n", model->config.num_transformer_layers);
    printf("config qkv_multipliers is: ");
    for (int i = 0; i < 2; i++) {
        printf("%f ", model->config.qkv_multipliers[i]);
    }
    printf("\n");

    printf("config rope_freq_constant is: %d\n", model->config.rope_freq_constant);
    printf("config rope_max_length is: %d\n", model->config.rope_max_length);
    printf("config vocab_size is: %d\n", model->config.vocab_size);
    printf("config max_qkv_proj_dim is: %d\n", model->config.max_qkv_proj_dim);
    printf("config max_intermediate_dim is: %d\n", model->config.max_intermediate_dim);

    size_t model_size = file_size - sizeof(model->config) - sizeof(int);
    model->num_parameters = model_size / sizeof(float);
    printf("num_parameters: %d\n", model->num_parameters);

    model->params_memory = (float*)malloc(model_size);
    fread(model->params_memory, sizeof(float), model->num_parameters, model_file);
    // for (int i = 0; i < 64; i++) {
    //     printf("weight: %f ", *(model->params_memory+i));
    // }
    // model->weights.token_embedding_table = model->params_memory;

    void *device_memory;
    cudaMalloc((void**)&device_memory, model_size);
    cudaMemcpy(device_memory, model->params_memory, model_size, cudaMemcpyHostToDevice);
    memory_map_weights(&model->weights, &model->config, (float*)device_memory);
}

typedef struct {

} Context;

typedef struct {
    int batch;
    int length;
    int* data;
} Prompt; 

void read_prompt(Prompt *prompt, const char* prompt_path) {
    FILE *prompt_file = fopen(prompt_path, "rb");
    if (prompt_file == NULL) {
        printf("Error opening prompt file\n");
    }

    int headers[2];
    fread(headers, sizeof(int), 2, prompt_file);
    prompt->batch = headers[0];
    prompt->length = headers[1];
    
    printf("prompt shape: %d %d\n", prompt->batch, prompt->length);

    prompt->data = (int*)malloc(prompt->batch * prompt->length * sizeof(float));
    fread(prompt->data, sizeof(float), prompt->batch * prompt->length, prompt_file);
    // for (int i = 0; i < prompt->batch * prompt->length; i++) {
    //     printf("%d ", *(prompt->data + i));
    // }
}

__device__ bool thread0() {
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
}

// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
__global__
void linear_forward(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
    
    int b = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;
    int kNThreads = blockDim.x;
    int out = bidy * kNThreads + tid;
    int offset_out = b * out_features + out;

    int offset_bias = out;
    float value = 0.0f;

    for (int in = 0; in < in_features; in++) {
        int offset_in = b * in_features + in;
        int offset_weight = out * in_features + in;                 
        value += input[offset_in] * weight[offset_weight];
    }

    output[offset_out] = value;
                
    if (bias != NULL) {
        output[offset_out] += bias[offset_bias];
    } 

    // if (thread0()) {
    //     printf("linear:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int i = 0; i < out_features; i++) {
    //             printf("%f, ", output[b * out_features + i]);
    //         }
    //         printf("]\n");
    //     }
    //     printf("]\n");
    // }
}

// // https://arxiv.org/pdf/1910.07467
// __global__
// void rmsnorm_forward(float* o, float* x, float *weight, int batch, int dim) {
//     // printf("rmsnorm_forward N:%d seq_len:%d dim:%d\n", batch, seq_len, dim);
//     // int b = 0;
//     // #pragma omp parallel for private(b)
//     int bidx = blockIdx.x; // batch
//     int bidy = blockIdx.y;
//     int tid = threadIdx.x; // thread id
//     int lid = tid % 32; // lane id
//     int wid = tid / 32; // warp id
//     int kWarp = blockDim.x / 32;

//     extern __shared__ float smem_[];
    
//     // 计算ss
//     // 
//     float ss = 0.0f;
//     int offset = bidx * dim;
//     #pragma unroll
//     for (int i = tid; i < dim; i += blockDim.x) {
//         ss += x[offset + i] * x[offset + i];
//     }

//     #pragma unroll
//     for (int mask = 32 / 2; mask > 0; mask /= 2) {
//         ss += __shfl_down_sync(uint32_t(-1), ss, mask);
//     }

//     if (lid == 0) {
//         int offset_warp = bidx * kWarp + wid;
//         smem_[offset_warp] = ss;
//     }

//     __syncthreads();

//     ss = 0.0f;
//     for (int i = 0; i < kWarp; i++) {
//         ss += smem_[bidx * kWarp + i];
//     }

//     ss /= dim;
//     ss += 1e-6f;
//     ss = 1.0f / sqrtf(ss);

//     int offset_x = bidx * dim + bidy * blockDim.x + tid;
//     int offset_w = bidy * blockDim.x + tid;
//     int offset_o = bidx * dim + bidy * blockDim.x + tid;
//     o[offset_o] = x[offset_x] * ss * weight[offset_w];

//     // if (thread0()) {
//     //     printf("rmsnorm:\n");
//     //     for (int b = 0; b < batch; b++) {
//     //         int offset = b * dim;
//     //         printf("[");
//     //         for (int d = 0; d < dim; d++) {
//     //              printf("%f, ", o[offset + d]);
//     //         }
//     //         printf("],\n");
//     //     }
//     // }
// }

// https://arxiv.org/pdf/1910.07467
__global__
void rmsnorm_forward(float* o, float* x, float *weight, int batch, int dim) {
    // printf("rmsnorm_forward N:%d seq_len:%d dim:%d\n", batch, seq_len, dim);
    // int b = 0;
    // #pragma omp parallel for private(b)
    int b = blockIdx.x; // batch
    int bidy = blockIdx.y;
    int tid = threadIdx.x; // thread id
    int lid = tid % 32; // lane id
    int wid = tid / 32; // warp id
    int kWarp = blockDim.x / 32;

    int kNThreads = blockDim.x;

    extern __shared__ float smem_[];
    
    // 计算ss
    // 
    int offset = b * dim;
            
    float ss = 0.0f;
    for (int d = 0; d < dim; d++) {
        ss += x[offset + d] * x[ offset + d];
    }
    ss /= dim;
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);
            
    for (int d = 0; d < dim; d++) {
            o[offset + d] = x[offset + d] * ss * weight[d];
     }

    // if (thread0()) {
    //     printf("rmsnorm:\n");
    //     for (int b = 0; b < batch; b++) {
    //         int offset = b * dim;
    //         printf("[");
    //         for (int d = 0; d < dim; d++) {
    //              printf("%f, ", o[offset + d]);
    //         }
    //         printf("],\n");
    //     }
    // }
}

__global__
void qkv_forward(float* input, float *q, float *qkv_proj, float *q_norm, float *k_norm, 
                          float *key_cache, float *value_cache, int qkv_proj_offset, int rope_freq_constant_,
                          int batch, int q_heads, int k_heads, int v_heads, int head_dim, int model_dim, int max_kv_heads, 
                          int max_seq_len, int num_transformer_layers, int layer_idx, int pos) {
    int b = blockIdx.x; // batch
    int bidy = blockIdx.y; 
    int tid = threadIdx.x; // thread id
    int kNThreads = blockDim.x;

    int out = bidy * kNThreads + tid;

    int offset_q = b * q_heads * head_dim;
    int offset_v = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                         + layer_idx * max_seq_len * max_kv_heads * head_dim 
                         + pos * max_kv_heads * head_dim;

            // int offset_bias = out;
    float value = 0.0f;

    for (int in = 0; in < model_dim; in++) {
        int offset_in = b * model_dim + in;
        int offset_weight = out * model_dim + in;                 
        value += input[offset_in] * (qkv_proj + qkv_proj_offset)[offset_weight];
    }

    if (out < q_heads * head_dim) {
        q[offset_q + out] = value;
    } else if (out < (q_heads + k_heads) * head_dim) {
        int offset_k = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                             + layer_idx * max_seq_len * max_kv_heads * head_dim 
                             + pos * max_kv_heads * head_dim;
                // if (offset_k + out - q_heads * head_dim == 0) {
                //     printf("batch:%d, num_transformer_layers:%d, max_seq_len:%d, kv_dim:%d, offset_k:%d pos:%d value=%f\n", batch, p->num_transformer_layers, s->max_seq_len, s->max_kv_heads * head_dim, offset_k, pos, value);
                // }
        key_cache[offset_k + out - q_heads * head_dim] = value;
    } else if (out < (q_heads + k_heads + v_heads) * head_dim) {
        value_cache[offset_v + out - (q_heads + k_heads) * head_dim] = value;
    }


    // if (thread0()) {
    // // printf qeury
    //     printf("query:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int h = 0; h < q_heads; h++) {
    //             printf("[");  
    //             int offset = b * q_heads * head_dim + h * head_dim;
    //             for (int hd = 0; hd < head_dim; hd++) {     
    //                 printf("%f,", q[offset + hd]);
    //             }
    //             printf("],\n");
    //         }
    //         printf("],\n");
    //     }

    //     // printf key
    //     printf("key:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int h = 0; h < k_heads; h++) {
    //             printf("[");
    //             int offset = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
    //                      + layer_idx * max_seq_len * max_kv_heads * head_dim 
    //                      + pos * max_kv_heads * head_dim
    //                      + h * head_dim;
    //             printf("offset=%d ", offset);
    //             for (int hd = 0; hd < head_dim; hd++) {
    //                 printf("%f,", key_cache[offset + hd]);
    //             }
    //             printf("],\n");
    //         }
    //         printf("],\n");
    //     }

    //     // printf value
    //     printf("value:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int h = 0; h < v_heads; h++) {
    //             printf("[");
    //             int offset = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
    //                      + layer_idx * max_seq_len * max_kv_heads * head_dim 
    //                       + pos * max_kv_heads * head_dim + h * head_dim;
    //             printf("offset=%d ", offset);
    //             for (int hd = 0; hd < head_dim; hd++) {
    //                 printf("%f,", value_cache[offset + hd]);
    //             }
    //             printf("],\n");
    //         }
    //         printf("],\n");
    //     }
    // }
}

// __global__ 
// void query_rope_forward(float *q, float *q_norm, int rope_freq_constant_, int batch, int q_heads, int head_dim, int layer_idx, int pos) {
//     float rope_freq_constant = (float)rope_freq_constant_;
    
//     int b = blockIdx.x;
//     int h = blockIdx.y;
//     int tid = threadIdx.x;
//     int lid = tid % 32;
//     int wid = tid / 32;

//     int offset = b * q_heads * head_dim + h * head_dim;

//     int kNThreads = blockDim.x;
//     int kWarps = kNThreads / 32;

//     extern __shared__ float smem_[];

//     float ss = 0.0f;
//     #pragma unroll
//     for (int i = tid; i < head_dim; i += kNThreads) {
//         ss += q[offset + i] * q[ offset + i];
//     }
    
//     #pragma unroll
//     for (int mask = 32 / 2; mask > 0; mask /= 2) {
//         ss += __shfl_down_sync(uint32_t(-1), ss, mask);
//     }

//     if (lid == 0) {
//         int offset_warp = b * kWarps + wid;
//         smem_[offset_warp] = ss;
//     }

//     __syncthreads();

//     ss = 0.0f;
//     for (int i = 0; i < kWarps; i++) {
//         ss += smem_[b * kWarps + i];
//     }

//     // __syncthreads();

//     ss /= head_dim;
//     ss += 1e-6f;
//     ss = 1.0f / sqrtf(ss);

//     for (int hd = tid; hd < head_dim; hd += kNThreads) {
//         q[offset + hd] = q[offset + hd] * ss * (q_norm + layer_idx * head_dim)[hd];
//     }

//     for (int hd = tid; hd < head_dim / 2; hd += kNThreads) {
//         float v0 = q[offset + hd];
//         float v1 = q[offset + hd + head_dim / 2];

//         float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
//         // printf("sl=%d %d=%f ", sl, hd, sl * freq);
//         float cos_val = cosf(pos * freq);
//         float sin_val = sinf(pos * freq);
//         // printf("sl=%d %d=%f ", sl, hd, sin_val);
//         q[offset + hd] = v0 * cos_val - v1 * sin_val;
//         q[offset + head_dim / 2 + hd] = v1 * cos_val + v0 * sin_val;
//                 // s->x_qkv_proj[offset + hd + head_dim / 2] = v0 * sin_val + v1 * cos_val;
//                 // printf("batch=%d seq_len=%d heads=%d %d=%f %f v=%f %f cos_sin=%f %f\n", b, sl, h, hd, s->x_qkv_proj[offset + hd], s->x_qkv_proj[offset + head_dim / 2 + hd], 
//                 //        v0, v1, cos_val, sin_val);

//                 // printf("batch=%d seq_len=%d heads=%d %d=%f %f\n", b, sl, h, hd, s->x_qkv_proj[offset + hd], s->x_qkv_proj[offset + head_dim / 2 + hd]);
//                 // printf("batch=%d seq_len=%d heads=%d %d=%f %f v=%f %f cos_sin=%f %f\n", b, sl, h, hd, s->x_qkv_proj[offset + hd], s->x_qkv_proj[offset + head_dim / 2 + hd], v0, v1, cos_val, sin_val);
//     }

//     // // printf query
//     // if (thread0()) {
//     //     for (int b = 0; b < batch; b++) {
//     //         printf("[");
//     //         for (int h = 0; h < q_heads; h++) {
//     //             printf("[");    
//     //             int offset = b * q_heads * head_dim + h * head_dim;
//     //             for (int hd = 0; hd < head_dim; hd++) {     
//     //                 printf("%f,", q[offset + hd]);
//     //             }
//     //             printf("],\n");
//     //         }
//     //         printf("],\n");
//     //     }
//     // }
// }

__global__ 
void query_rope_forward(float *q, float *q_norm, int rope_freq_constant_, int batch, int q_heads, int head_dim, int layer_idx, int pos) {
    float rope_freq_constant = (float)rope_freq_constant_;
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int lid = tid % 32;
    int wid = tid / 32;

    int offset = b * q_heads * head_dim + h * head_dim;

    int kNThreads = blockDim.x;
    int kWarps = kNThreads / 32;

    extern __shared__ float smem_[];

    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < head_dim; i++) {
        ss += q[offset + i] * q[ offset + i];
    }

    ss /= head_dim;
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);

    for (int hd = 0; hd < head_dim; hd++) {
        q[offset + hd] = q[offset + hd] * ss * (q_norm + layer_idx * head_dim)[hd];
    }

    for (int hd = 0; hd < head_dim / 2; hd++) {
        float v0 = q[offset + hd];
        float v1 = q[offset + hd + head_dim / 2];

        float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
        // printf("sl=%d %d=%f ", sl, hd, sl * freq);
        float cos_val = cosf(pos * freq);
        float sin_val = sinf(pos * freq);
        // printf("sl=%d %d=%f ", sl, hd, sin_val);
        q[offset + hd] = v0 * cos_val - v1 * sin_val;
        q[offset + head_dim / 2 + hd] = v1 * cos_val + v0 * sin_val;
                // s->x_qkv_proj[offset + hd + head_dim / 2] = v0 * sin_val + v1 * cos_val;
                // printf("batch=%d seq_len=%d heads=%d %d=%f %f v=%f %f cos_sin=%f %f\n", b, sl, h, hd, s->x_qkv_proj[offset + hd], s->x_qkv_proj[offset + head_dim / 2 + hd], 
                //        v0, v1, cos_val, sin_val);

                // printf("batch=%d seq_len=%d heads=%d %d=%f %f\n", b, sl, h, hd, s->x_qkv_proj[offset + hd], s->x_qkv_proj[offset + head_dim / 2 + hd]);
                // printf("batch=%d seq_len=%d heads=%d %d=%f %f v=%f %f cos_sin=%f %f\n", b, sl, h, hd, s->x_qkv_proj[offset + hd], s->x_qkv_proj[offset + head_dim / 2 + hd], v0, v1, cos_val, sin_val);
    }

    // // printf query
    // if (thread0()) {
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int h = 0; h < q_heads; h++) {
    //             printf("[");    
    //             int offset = b * q_heads * head_dim + h * head_dim;
    //             for (int hd = 0; hd < head_dim; hd++) {     
    //                 printf("%f,", q[offset + hd]);
    //             }
    //             printf("],\n");
    //         }
    //         printf("],\n");
    //     }
    // }
}


// __global__ 
// void key_rope_forward(float *key_cache, float *k_norm, int rope_freq_constant_, 
//                       int batch, int k_heads, int head_dim, int max_kv_heads, 
//                       int max_seq_len, int num_transformer_layers, int layer_idx, int pos) {
//     float rope_freq_constant = (float)rope_freq_constant_;
    
//     int b = blockIdx.x;
//     int h = blockIdx.y;
//     int tid = threadIdx.x;
//     int lid = tid % 32;
//     int wid = tid / 32;

//     int offset = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
//                          + layer_idx * max_seq_len * max_kv_heads * head_dim  + pos * max_kv_heads * head_dim + h * head_dim;

//     int kNThreads = blockDim.x;
//     int kWarps = kNThreads / 32;

//     extern __shared__ float smem_[];

//     float ss = 0.0f;
//     #pragma unroll
//     for (int i = tid; i < head_dim; i += kNThreads) {
//         ss += key_cache[offset + i] * key_cache[ offset + i];
//     }
    
//     #pragma unroll
//     for (int mask = 32 / 2; mask > 0; mask /= 2) {
//         ss += __shfl_down_sync(uint32_t(-1), ss, mask);
//     }

//     if (lid == 0) {
//         int offset_warp = b * kWarps + wid;
//         smem_[offset_warp] = ss;
//     }

//     __syncthreads();

//     ss = 0.0f;
//     for (int i = 0; i < kWarps; i++) {
//         ss += smem_[b * kWarps + i];
//     }

//     // __syncthreads();

//     ss /= head_dim;
//     ss += 1e-6f;
//     ss = 1.0f / sqrtf(ss);

//     for (int hd = tid; hd < head_dim; hd += kNThreads) {
//         key_cache[offset + hd] = key_cache[offset + hd] * ss * (k_norm  + layer_idx * head_dim)[hd];
//     }

//     for (int hd = tid; hd < head_dim / 2; hd += kNThreads) {
//         float v0 = key_cache[offset + hd];
//         float v1 = key_cache[offset + hd + head_dim / 2];

//         float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
//         // printf("sl=%d %d=%f ", sl, hd, sl * freq);
//         float cos_val = cosf(pos * freq);
//         float sin_val = sinf(pos * freq);
//                 // printf("sl=%d %d=%f ", sl, hd, sin_val);
//         key_cache[offset + hd] = v0 * cos_val - v1 * sin_val;
//         key_cache[offset + head_dim / 2 + hd] = v1 * cos_val + v0 * sin_val;
//     }

//     // if (thread0()) {
//     //     for (int b = 0; b < batch; b++) {
//     //         printf("[");
//     //         for (int h = 0; h < k_heads; h++) {
//     //             printf("[");
//     //             int offset = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
//     //                      + layer_idx * max_seq_len * max_kv_heads * head_dim  + 0 * max_kv_heads * head_dim
//     //                      + h * head_dim;
//     //             printf("offset=%d ", offset);
//     //             for (int hd = 0; hd < head_dim; hd++) {
//     //                 printf("%f,", key_cache[offset + hd]);
//     //             }
//     //             printf("],\n");
//     //         }
//     //         printf("],\n");
//     //     }
//     // }
// }


__global__ 
void key_rope_forward(float *key_cache, float *k_norm, int rope_freq_constant_, 
                      int batch, int k_heads, int head_dim, int max_kv_heads, 
                      int max_seq_len, int num_transformer_layers, int layer_idx, int pos) {
    float rope_freq_constant = (float)rope_freq_constant_;
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int lid = tid % 32;
    int wid = tid / 32;

    int offset = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                         + layer_idx * max_seq_len * max_kv_heads * head_dim  + pos * max_kv_heads * head_dim + h * head_dim;

    int kNThreads = blockDim.x;
    int kWarps = kNThreads / 32;

    extern __shared__ float smem_[];

    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < head_dim; i++) {
        ss += key_cache[offset + i] * key_cache[ offset + i];
    }

    ss /= head_dim;
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);

    for (int hd = 0; hd < head_dim; hd++) {
        key_cache[offset + hd] = key_cache[offset + hd] * ss * (k_norm  + layer_idx * head_dim)[hd];
    }

    for (int hd = 0; hd < head_dim / 2; hd++) {
        float v0 = key_cache[offset + hd];
        float v1 = key_cache[offset + hd + head_dim / 2];

        float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
        // printf("sl=%d %d=%f ", sl, hd, sl * freq);
        float cos_val = cosf(pos * freq);
        float sin_val = sinf(pos * freq);
                // printf("sl=%d %d=%f ", sl, hd, sin_val);
        key_cache[offset + hd] = v0 * cos_val - v1 * sin_val;
        key_cache[offset + head_dim / 2 + hd] = v1 * cos_val + v0 * sin_val;
    }

    // if (thread0()) {
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int h = 0; h < k_heads; h++) {
    //             printf("[");
    //             int offset = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
    //                      + layer_idx * max_seq_len * max_kv_heads * head_dim  + 0 * max_kv_heads * head_dim
    //                      + h * head_dim;
    //             printf("offset=%d ", offset);
    //             for (int hd = 0; hd < head_dim; hd++) {
    //                 printf("%f,", key_cache[offset + hd]);
    //             }
    //             printf("],\n");
    //         }
    //         printf("],\n");
    //     }
    // }
}


__global__
void rmsnorm_rope_forward(float* input, float *q, float *qkv_proj, float *q_norm, float *k_norm, 
                          float *key_cache, float *value_cache, int qkv_proj_offset, int rope_freq_constant_,
                          int batch, int q_heads, int k_heads, int v_heads, int head_dim, int model_dim, int max_kv_heads, 
                          int max_seq_len, int num_transformer_layers, int layer_idx, int pos) {

}

// __global__
// void group_attention_forward(float* output, float *q, float *key_cache, float *value_cache, float *att,
//                              int batch, int q_heads, int k_heads, int head_dim, int max_q_heads, int max_kv_heads, int max_seq_len, 
//                              int num_transformer_layers, int layer_idx, int pos) {
//     int num_groups = q_heads / k_heads;
//     int b = blockIdx.x;
//     int h = blockIdx.y;
//     int tid = threadIdx.x;
//     int kNThreads = blockDim.x;
//     int kWarps = kNThreads / 32;
//     int lid = tid % 32; // lane id
//     int wid = tid / 32; // warp id
//     extern __shared__ float smem_[];

//     int offset_att = b * max_q_heads * max_seq_len + h * max_seq_len;
//     int offset_q = b * q_heads * head_dim + h * head_dim;

    
//     for (int lk = tid; lk < pos + 1; lk += kNThreads) {
//         int offset_k = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
//                          + layer_idx * max_seq_len * max_kv_heads * head_dim 
//                          + lk * max_kv_heads * head_dim
//                          + (h / num_groups)  * head_dim;

//         float score = 0.0f;
//         for (int i = 0; i < head_dim; i++) {
//             score += q[offset_q + i] * key_cache[offset_k + i];
//                         // if (h == 0 && lk == 0) {
//                         //     printf("offset_k:%d batch:%d, i:%d, q:%f, k:%f\n", offset_k, b, i, s->q[offset_q+i], s->key_cache[offset_k + i]);
//                         // }
//         }

//         score /= sqrtf((float)head_dim);
//         att[offset_att + lk] = score;
//                 // printf("%f ", score);
//     }

//                 // printf("\n");
//     float max_val = att[offset_att];
//     for (int lk = tid; lk < pos + 1; lk += kNThreads) { 
//         if (att[offset_att + lk] > max_val) {
//             max_val = att[offset_att + lk];
//         }
//     }

//     #pragma unroll
//     for (int mask = 32 / 2; mask > 0; mask /= 2) {
//         float shfl_max = __shfl_down_sync(uint32_t(-1), max_val, mask);
//         if (shfl_max > max_val) {
//             max_val = shfl_max;
//         }
//     }

//     if (lid == 0) {
//         int offset_warp = b * kWarps + wid;
//         smem_[offset_warp] = max_val;
//     }

//     __syncthreads();

//     for (int i = 0; i < kWarps; i++) {
//         if (max_val < smem_[b * kWarps + i]) {
//             max_val = smem_[b * kWarps + i];
//         }
//     }

//     // __syncthreads();

//     float ss = 0.0f;
//     #pragma unroll
//     for (int lk = tid; lk < pos + 1; lk += kNThreads) {
//         ss += expf(att[offset_att + lk] - max_val);
//     }
    
//     #pragma unroll
//     for (int mask = 32 / 2; mask > 0; mask /= 2) {
//         ss += __shfl_down_sync(uint32_t(-1), ss, mask);
//     }

//     if (lid == 0) {
//         int offset_warp = b * kWarps + wid;
//         smem_[offset_warp] = ss;
//     }

//     __syncthreads();

//     ss = 0.0f;
//     for (int i = 0; i < kWarps; i++) {
//         ss += smem_[b * kWarps + i];
//     }

//     for (int lk = tid; lk < pos + 1; lk += kNThreads) { 
//         att[offset_att + lk] = expf(att[offset_att + lk] - max_val) / ss;
//     }

//     int offset_o = b * q_heads * head_dim + h * head_dim;

//     for (int lv = tid; lv < head_dim; lv += kNThreads){
//         float sv = 0.0f;
//         for (int k = 0; k < pos + 1; k++) { 
//             int offset_v = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
//                          + layer_idx * max_seq_len * max_kv_heads * head_dim 
//                          + k * max_kv_heads * head_dim
//                          + (h / num_groups) * head_dim;
//             sv += att[offset_att + k] * (value_cache[offset_v + lv]);
//         }
//         output[offset_o + lv] = sv;
//     }

//     // if (thread0()) {
//     //     printf("group_attention:\n");
//     //     for (int b = 0; b < batch; b++) {
//     //         printf("[");
//     //         for (int d = 0; d < q_heads * head_dim; d++) {
//     //             int offset = b * q_heads * head_dim;
//     //                 printf("%f, ",output[offset + d]);
//     //         }
//     //         printf("],\n");
//     //     }
//     // }
// }

__global__
void group_attention_forward(float* output, float *q, float *key_cache, float *value_cache, float *att,
                             int batch, int q_heads, int k_heads, int head_dim, int max_q_heads, int max_kv_heads, int max_seq_len, 
                             int num_transformer_layers, int layer_idx, int pos) {
    int num_groups = q_heads / k_heads;
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int kNThreads = blockDim.x;
    int kWarps = kNThreads / 32;
    int lid = tid % 32; // lane id
    int wid = tid / 32; // warp id
    extern __shared__ float smem_[];

    int offset_att = b * max_q_heads * max_seq_len + h * max_seq_len;
    int offset_q = b * q_heads * head_dim + h * head_dim;

    
    for (int lk = 0; lk < pos + 1; lk++) {
        int offset_k = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                         + layer_idx * max_seq_len * max_kv_heads * head_dim 
                         + lk * max_kv_heads * head_dim
                         + (h / num_groups)  * head_dim;

        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += q[offset_q + i] * key_cache[offset_k + i];
                        // if (h == 0 && lk == 0) {
                        //     printf("offset_k:%d batch:%d, i:%d, q:%f, k:%f\n", offset_k, b, i, s->q[offset_q+i], s->key_cache[offset_k + i]);
                        // }
        }

        score /= sqrtf((float)head_dim);
        att[offset_att + lk] = score;
                // printf("%f ", score);
    }

                // printf("\n");
    float max_val = att[offset_att];
    for (int lk = 0; lk < pos + 1; lk++) { 
        if (att[offset_att + lk] > max_val) {
            max_val = att[offset_att + lk];
        }
    }

    float ss = 0.0f;
    #pragma unroll
    for (int lk = 0; lk < pos + 1; lk++) {
        ss += expf(att[offset_att + lk] - max_val);
    }

    for (int lk = 0; lk < pos + 1; lk++) { 
        att[offset_att + lk] = expf(att[offset_att + lk] - max_val) / ss;
    }

    int offset_o = b * q_heads * head_dim + h * head_dim;

    for (int lv = 0; lv < head_dim; lv++){
        float sv = 0.0f;
        for (int k = 0; k < pos + 1; k++) { 
            int offset_v = b * num_transformer_layers * max_seq_len * max_kv_heads * head_dim 
                         + layer_idx * max_seq_len * max_kv_heads * head_dim 
                         + k * max_kv_heads * head_dim
                         + (h / num_groups) * head_dim;
            sv += att[offset_att + k] * (value_cache[offset_v + lv]);
        }
        output[offset_o + lv] = sv;
    }

    // if (thread0()) {
    //     printf("group_attention:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int d = 0; d < q_heads * head_dim; d++) {
    //             int offset = b * q_heads * head_dim;
    //                 printf("%f, ",output[offset + d]);
    //         }
    //         printf("],\n");
    //     }
    // }
}

__global__
void residual_forward(float *x, float *xb, int batch, int dim) {
    int b = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;
    int kNThreads = blockDim.x;
    int offset = b * dim + bidy * kNThreads + tid;

    x[offset] += xb[offset];

    // if (thread0()) {
    //     printf("residual:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int i = 0; i < dim; i++) {
    //             int offset_x = b * dim + i;
    //             printf("%f, ", x[offset_x]);
    //         }
    //         printf("]\n");
    //     }
    // }
}

int make_divisible(float v, int divisor, int min_value) {
    if (min_value < 0) {
        min_value = divisor;
    }
    int new_v = (int)(v + (float)divisor / 2.0f) / divisor * divisor;
    if (min_value > new_v) {
        new_v = min_value;
    }
    return new_v;
}

__global__
void glu_forward(float *ihb2, float* ihb, int batch, int intermediate_dim, int layer_idx) {

    int b = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;
    int kNThreads = blockDim.x;

    int offset_y1 = b * 2 * intermediate_dim + bidy * kNThreads + tid;
    int offset_y2 = b * 2 * intermediate_dim + intermediate_dim + bidy * kNThreads + tid;
    int offset_h = b * intermediate_dim + bidy * kNThreads + tid;

    ihb[offset_y1] = ihb[offset_y1] / (1 + expf(-(ihb[offset_y1])));
    ihb2[offset_h] = ihb[offset_y1] * ihb[offset_y2];

    // if (thread0()) {
    //     printf("glu:\n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int i = 0; i < intermediate_dim; i++) {
    //             printf("%f, ", ihb2[b * intermediate_dim + i]);
    //         }
    //         printf("]\n");
    //     }
    // }
}

__global__
void logits_forward(float* output, float* input, float *weight, float* bias, int batch, int in_features, int out_features) {
    int b = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;
    int kNThreads = blockDim.x;

    int out = bidy * kNThreads + tid;
    int offset_out = b * out_features + out;
    int offset_bias = out;
    float value = 0.0f;
    for (int in = 0; in < in_features; in++) {
        int offset_in = b * in_features + in;
        int offset_weight = out * in_features + in;
        value += input[offset_in] * weight[offset_weight];
    }
    output[offset_out] = value;
    if (bias != NULL) {
        output[offset_out] += bias[offset_bias];
    } 

    // if (thread0()) {
    //     printf("logits: \n");
    //     for (int b = 0; b < batch; b++) {
    //         printf("[");
    //         for (int i = 0; i < out_features; i++) {
    //             printf("%f, ", output[b * out_features + i]);
    //         }
    //         printf("]\n");
    //     }
    // }
}

// __global__
// void argmax_forward(int* output, float* input, int batch, int dim) {
//     int b = blockIdx.x;
//     // int bidy = blockIdx.y;
//     int tid = threadIdx.x;
//     int kNThreads = blockDim.x;
//     int kWarps = kNThreads / 32;
//     int lid = tid % 32; // lane id
//     int wid = tid / 32; // warp id
//     // extern __shared__ int ismem_[];
//     __shared__ int ismem_[8];

//     int offset = b * dim;

//     int max_i = tid;
//     float max_val = input[offset + max_i];
    
    
//     for (int i = tid; i < dim; i += kNThreads) { 
//         if (input[offset + i] > max_val) {
//             max_val = input[offset + i];
//             max_i = i;
//         }
//     }

//     #pragma unroll
//     for (int mask = 32 / 2; mask > 0; mask /= 2) {
//         int shfl_i = __shfl_down_sync(uint32_t(-1), max_i, mask);
//         if (input[offset + shfl_i] > max_val) {
//             max_val = input[offset + shfl_i];
//             max_i = shfl_i;
//         }
//     }

//     if (lid == 0) {
//         int offset_warp = b * kWarps + wid;
//         ismem_[offset_warp] = max_i;
//         // printf("offset_warp: %d %d\n", offset_warp, ismem_[offset_warp]);
//     }

//     __syncthreads();
//     // __syncwarp();
 
//     // if (thread0() && b == 1) {
//     //     printf("ismem_: \n");
//     //     for (int b = 0; b < batch; b++) {
//     //         printf("[");
//     //         for (int i = 0; i < kWarps; i++) {
//     //             printf("%d, ", ismem_[b * kWarps + i]);
//     //         }
//     //         printf("]\n");
//     //     }
//     // }
    
//     for (int i = b * kWarps; i < (b + 1) * kWarps; i++) {
//         if (max_val < input[offset + ismem_[i]]) {
//             max_val = input[offset + ismem_[i]];
//             max_i = ismem_[i];
//         }
//     }

//     __syncthreads();
//     output[b] = max_i;

//     if (thread0()) {
//         printf("argmax:\n");
//         printf("[");
//         for (int b = 0; b < batch; b++) {
//             printf("%d, ", output[b]);
//         }
//         printf("]\n");
//     }
// }

__global__
void argmax_forward(int* output, float* input, int batch, int dim) {
    int b = blockIdx.x;

    int v = 0;
    for (int n = 1; n < dim; n++) {
        if (input[b * dim + n] > input[b * dim + v]) {
               v = n;
           }
        }
    output[b] = v;

    __syncthreads();
    // if (thread0()) {
    //     printf("argmax:\n");
    //     printf("[");
    //     for (int b = 0; b < batch; b++) {
    //         printf("%d, ", output[b]);
    //     }
    //     printf("]\n");
    // }
}


__global__ void get_content_row(float *x, float* token_embeddings, int *token, int batch, int dim) {
    int bidx = blockIdx.x; // batch
    int bidy = blockIdx.y; // dim = 
    int tidx = threadIdx.x;
    int offset_x = bidx * dim + bidy * blockDim.x + tidx;
    int offset_t = bidy * blockDim.x + tidx;
    x[offset_x] = *(token_embeddings + token[bidx] * dim + offset_t);

    // if (thread0()) {
    //     // printf("+++++%d %d %d\n", blockDim.x, gridDim.x, gridDim.y);
    //     printf("[");
    //     for (int b = 0; b < batch; b++) {
    //         int offset_x = b * dim;
    //         printf("[");
    //         for (int i = 0; i<dim; i++) {
    //             printf("%f, ", x[offset_x + i]);
    //         }
    //         printf("],\n");
    //     }
    //     printf("]\n");
    // }
}

float* openelm_forward(Context *ctx, OpenELM* openelm, int *token, int batch, int pos) {
    OpenELMConfig *p = &openelm->config;
    OpenELMWeights *w = &openelm->weights;
    RunState* s = &openelm->state;

    s->qkv_proj_offset = 0;
    s->out_proj_offset = 0;
    s->proj_1_offset = 0;
    s->proj_2_offset = 0;
    float *x = s->x;
    int model_dim = p->model_dim;
    int head_dim = p->head_dim;
    int max_q_heads = s->max_q_heads;
    int max_seq_len = s->max_seq_len;
    int max_kv_heads = s->max_kv_heads;
    int num_transformer_layers = p->num_transformer_layers;
    float rope_freq_constant = (float)p->rope_freq_constant;

    // printf("openelm_forward pos:%d, batch:%d, model_dim:%d \n", pos, batch, model_dim);
    int kNThreads = 128;
    
    get_content_row<<<dim3(batch, model_dim/kNThreads), kNThreads>>>(x, w->token_embeddings, token, batch, model_dim);
    

    // for(int l = 0; l < 1; l++) {
    for(int l = 0; l < p->num_transformer_layers; l++) {
        // attn_norm
        // rmsnorm_forward<<<dim3(batch, model_dim / kNThreads), kNThreads>>>(s->xb, s->x, w->attn_norm + l*model_dim, batch, model_dim);
        rmsnorm_forward<<<dim3(batch), 1>>>(s->xb, s->x, w->attn_norm + l*model_dim, batch, model_dim);
        
        // MultiHeadCausalAttention
        int q_heads = p->num_query_heads[l];
        int k_heads = p->num_kv_heads[l];
        int v_heads = p->num_kv_heads[l];
        int out_features = (q_heads + k_heads + v_heads) * head_dim;
        //
        qkv_forward<<<dim3(batch, out_features / kNThreads), kNThreads>>>(s->xb, s->q, w->qkv_proj, w->q_norm, w->k_norm, 
                          s->key_cache, s->value_cache, s->qkv_proj_offset, p->rope_freq_constant,
                          batch, q_heads, k_heads, v_heads, head_dim, model_dim, max_kv_heads, s->max_seq_len, p->num_transformer_layers, l, pos);
        s->qkv_proj_offset += out_features * model_dim;
        
        // rope
        // query_rope_forward<<<dim3(batch, q_heads), kNThreads>>>(s->q, w->q_norm, p->rope_freq_constant, batch, q_heads, head_dim, l, pos);
        query_rope_forward<<<dim3(batch, q_heads), 1>>>(s->q, w->q_norm, p->rope_freq_constant, batch, q_heads, head_dim, l, pos);


        // key_rope_forward<<<dim3(batch, k_heads), kNThreads>>>(s->key_cache, w->k_norm, p->rope_freq_constant, 
        //               batch, k_heads, head_dim, max_kv_heads, 
        //               s->max_seq_len, p->num_transformer_layers, l, pos);
        key_rope_forward<<<dim3(batch, k_heads), 1>>>(s->key_cache, w->k_norm, p->rope_freq_constant, 
                      batch, k_heads, head_dim, max_kv_heads, 
                      s->max_seq_len, p->num_transformer_layers, l, pos);

        // group attention
        // group_attention_forward<<<dim3(batch, q_heads),kNThreads>>>(s->xb, s->q, s->key_cache, s->value_cache, s->att,
        //                      batch, q_heads, k_heads, head_dim, max_q_heads, max_kv_heads, max_seq_len, 
        //                      num_transformer_layers, l, pos);
        group_attention_forward<<<dim3(batch, q_heads),1>>>(s->xb, s->q, s->key_cache, s->value_cache, s->att,
                             batch, q_heads, k_heads, head_dim, max_q_heads, max_kv_heads, max_seq_len, 
                             num_transformer_layers, l, pos);

        
        linear_forward<<<dim3(batch, model_dim / kNThreads), kNThreads>>>(s->xb2, s->xb, w->out_proj + s->out_proj_offset, NULL, batch, q_heads * head_dim, model_dim);
        s->out_proj_offset += q_heads * head_dim * model_dim;

        residual_forward<<<dim3(batch, model_dim / kNThreads), kNThreads>>>(s->x, s->xb2, batch, model_dim);

        // ffn_norm
        // rmsnorm_forward<<<dim3(batch, model_dim / kNThreads), kNThreads>>>(s->xb, s->x, w->ffn_norm + l*model_dim, batch, model_dim);
        rmsnorm_forward<<<dim3(batch), 1>>>(s->xb, s->x, w->ffn_norm + l*model_dim, batch, model_dim);

        // FeedForwardNetwork
        int intermediate_dim = (int)make_divisible(p->ffn_multipliers[l] * model_dim, p->ffn_dim_divisor, -1);

        linear_forward<<<dim3(batch, 2 * intermediate_dim / kNThreads), kNThreads>>>(s->ihb, s->xb, w->proj_1 + s->proj_1_offset, NULL, batch, model_dim, 2 * intermediate_dim);
        s->proj_1_offset += 2 * intermediate_dim * model_dim;

        glu_forward<<<dim3(batch, intermediate_dim / kNThreads), kNThreads>>>(s->ihb2, s->ihb, batch, intermediate_dim, l);

        linear_forward<<<dim3(batch, model_dim / kNThreads), kNThreads>>>(s->hb, s->ihb2, w->proj_2 + s->proj_2_offset, NULL, batch, intermediate_dim, model_dim);
        s->proj_2_offset += intermediate_dim * model_dim;

        residual_forward<<<dim3(batch, model_dim / kNThreads), kNThreads>>>(s->x, s->hb, batch, model_dim);
    }

    // rmsnorm_forward<<<dim3(batch, model_dim / kNThreads), kNThreads>>>(s->x, s->x, w->norm, batch, model_dim);
    rmsnorm_forward<<<dim3(batch), 1>>>(s->x, s->x, w->norm, batch, model_dim);

    logits_forward<<<dim3(batch, p->vocab_size / kNThreads), kNThreads>>>(s->logits, s->x, w->token_embeddings, NULL, batch, model_dim, p->vocab_size);
    // cudaDeviceSynchronize();

    // logits_forward(s->logits, s->x, w->token_embeddings, NULL, batch, 1, model_dim, p->vocab_size);

    // // for (int i = 0; i < batch * p->vocab_size; i++) {
    // //     printf("%d=%f ", i, s->logits[i]);
    // // }

    return s->logits;
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(Context *ctx, OpenELM *openelm, Prompt *prompt, int steps) {
    RunState *s = &openelm->state;
    for (int b = 0; b < prompt->batch; b++) {
        printf("the %d prompts: ", b);
        int *data = prompt->data + b * prompt->length;
        for (int l = 0; l < prompt->length; l++) {
            printf("%d ", data[l]);
        }
        printf("\n");
    }

    // int num_prompt_tokens = prompt->length;
    // int* prompt_tokens = prompt->data;

    int pos = 0;
    
    for (int i = 0; i < prompt->batch; i++) {
        int offset = i * prompt->length + pos;
        cudaMemcpy(s->token + i, prompt->data + offset, sizeof(int), cudaMemcpyHostToDevice);
    }

    long start = time_in_ms();
    while (pos < steps) {
    // while (pos < steps) {
        // float *logits = 
        openelm_forward(ctx, openelm, s->token, prompt->batch, pos);

        // argmax_forward<<<dim3(s->batch, openelm->config.vocab_size / 128), 128>>>(s->next, s->logits, s->batch, openelm->config.vocab_size);
        // argmax_forward(s->next, s->logits, s->batch, openelm->config.vocab_size);
        argmax_forward<<<s->batch, 1>>>(s->next, s->logits, s->batch, openelm->config.vocab_size);

        cudaDeviceSynchronize();

        pos++;

        if (pos < prompt->length) {
            for (int i = 0; i < s->batch; i++) {
                int offset = i * prompt->length + pos ;
                // s->next[i] = prompt->data[offset];
                cudaMemcpy(s->next + i, prompt->data + offset, sizeof(int), cudaMemcpyHostToDevice);
            }
        }

        for (int i = 0; i < s->batch; i++) {
            cudaMemcpy(s->token + i, s->next + i, sizeof(int), cudaMemcpyDeviceToDevice);
        }

        for (int i = 0; i < s->batch; i++) {
            cudaMemcpy(s->next_cpu + i, s->next + i, sizeof(int), cudaMemcpyDeviceToHost);
        }

        printf("pos:%d ", pos);
        for (int i = 0; i < s->batch; i++) {
            printf("%d ", s->next_cpu[i]);
        }
        printf("\n");

    }
    long end = time_in_ms();
    fprintf(stderr, "seconds:%fs tokens:%d achieved tok/s: %f\n", (double)(end-start) / 1000, pos, (pos) / (double)(end-start)*1000);
}


OpenELM py_model;

void c_init(int batch, int max_seq_len) {
    openelm_build_from_checkpoint(&py_model, "openelm_270M.bin");
    py_model.state.batch = batch;
    py_model.state.max_seq_len = max_seq_len;
    malloc_run_state(&py_model.state, &py_model.config);
}

// void get_mod
int* c_openelm_forward(int batch, int seq_len, int *data, int pos) {
    printf("c_openelm_forward batch:%d, seq_len:%d, pos:%d\n", batch, seq_len, pos);
    RunState *s = &py_model.state;
    
    // int* prompt_tokens = data;
    // int start = 0;
    for (int i = 0; i < batch; i++) {
        // s->token[i] = data[i];
        cudaMemcpy(s->token + i, data + i, sizeof(int), cudaMemcpyHostToDevice);
    }
    
    Context ctx;
    openelm_forward(&ctx, &py_model, s->token, batch, pos);
    argmax_forward<<<s->batch, 1>>>(s->next, s->logits, s->batch, py_model.config.vocab_size);
    cudaDeviceSynchronize();

    for (int i = 0; i < s->batch; i++) {
        cudaMemcpy(s->next_cpu + i, s->next + i, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // printf("pos:%d ", pos);
    // for (int i = 0; i < s->batch; i++) {
    //     printf("%d ", s->next_cpu[i]);
    // }
    // printf("\n");
    return s->next_cpu;
}

void c_generate(int batch, int seq_len, int *data, int steps) {
    Context ctx;
    Prompt prompt;
    prompt.batch = batch;
    prompt.length = seq_len;
    prompt.data = data;
    generate(&ctx, &py_model, &prompt, steps);
    printf("hello openelm\n");
}

void c_chat () {

}

int start() {
    Prompt prompt;
    read_prompt(&prompt, "openelm_prompt.bin");
    prompt.batch = 2;

    OpenELM model;
    // OpenELMConfig config;
    openelm_build_from_checkpoint(&model, "openelm_270M.bin");
    model.state.batch = prompt.batch;
    model.state.max_seq_len = 256;
    malloc_run_state(&model.state, &model.config);
    
    Context ctx;
    generate(&ctx, &model, &prompt, 256);
    printf("hello openelm\n");
    return 0;
}

int main(int argc, char** argv) {
    return start();
}