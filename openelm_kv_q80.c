/*
gcc -o openelm_kv_q80 -g  openelm_kv_q80.c -lm -fopenmp
gcc --shared -fPIC -o openelm_kv_q80.so openelm_kv_q80.c -lm -fopenmp

gcc -o openelm_kv_q80 -g -O3 openelm_kv_q80.c -lm -fopenmp

*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// extern "C" int start();

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
    int8_t* q;    // quantized values
    float* s;     // scaling factors
} QuantizedTensor;

typedef struct {
    float *token_embeddings; // transformer.token_embeddings.weight
    float *attn_norm;        // transformer.layers.{i}.attn_norm.weight
    QuantizedTensor *qkv_proj;         // transformer.layers.{i}.attn.qkv_proj.weight
    float *q_norm;           // transformer.layers.{i}.attn.q_norm.weight
    float *k_norm;           // transformer.layers.{i}.attn.k_norm.weight
    QuantizedTensor *out_proj;         // transformer.layers.{i}.attn.out_proj.weight
    QuantizedTensor *proj_1;           // transformer.layers.{i}.ffn.proj_1.weight
    QuantizedTensor *proj_2;           // transformer.layers.{i}.ffn.proj_2.weight
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
    int *sample;
    int *next;
    int *token;

    float *q;
    float *key_cache;
    float *value_cache;

    int batch;
    int seq_len;
    int max_seq_len;

    int qkv_proj_offset_q;
    int qkv_proj_offset_s;
    int out_proj_offset_q;
    int out_proj_offset_s;
    int proj_1_offset_q;
    int proj_1_offset_s;
    int proj_2_offset_q;
    int proj_2_offset_s;

    int max_q_heads;
    int max_kv_heads;
} RunState;

typedef struct {
    OpenELMConfig config;
    RunState state;
    OpenELMWeights *weights;
    int num_parameters;
    char *params_memory;
} OpenELM;

void malloc_run_state(RunState* s, OpenELMConfig* p) {
    // printf("+++%d %d", s->batch, p->model_dim);
    int seq_len = s->max_seq_len;
    s->x = (float*)malloc(s->batch * p->model_dim * sizeof(float));
    // printf("+++%d %d", s->batch, p->max_qkv_proj_dim);
    // s->x_qkv_proj = (float*)malloc(s->batch * seq_len * p->max_qkv_proj_dim * sizeof(float));
    s->xb2 = (float*)malloc(s->batch * p->model_dim * sizeof(float));
    
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
    s->xb = (float*)malloc(s->batch * max_xb_dim * sizeof(float));

    s->att = (float*)malloc(s->batch * q_heads * seq_len  * sizeof(float));
    s->ihb = (float*)malloc(s->batch * seq_len * 2 * p->max_intermediate_dim * sizeof(float));
    s->ihb2 = (float*)malloc(s->batch * seq_len * p->max_intermediate_dim * sizeof(float));

    s->hb = (float*)malloc(s->batch * seq_len * p->model_dim * sizeof(float));
    // s->ihb2 = (float*)malloc(s->batch * seq_len * p->max_intermediate_dim * sizeof(float));
    s->logits = (float*)malloc(s->batch * seq_len * p->vocab_size * sizeof(float));
    s->sample = (int*)malloc(s->batch * sizeof(int));
    s->next = (int*)malloc(s->batch * sizeof(int));
    s->token = (int*)malloc(s->batch * sizeof(int));
    s->q = (float*)malloc(s->batch * q_heads * p->head_dim * sizeof(float));
    s->key_cache = (float*)malloc(s->batch * p->num_transformer_layers * seq_len * k_heads * p->head_dim * sizeof(float));
    s->value_cache = (float*)malloc(s->batch * p->num_transformer_layers * seq_len * v_heads * p->head_dim * sizeof(float));
}

void free_run_state(RunState* s) {

}

void memory_map_weights(OpenELMWeights *w, OpenELMConfig* p, char* ptr) {
    int ll;
    int ll_s;
    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    w->token_embeddings = (float*)ptr;
    ptr += ll * sizeof(float);

    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    w->attn_norm = (float*)ptr;
    ptr += ll * sizeof(float);

    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    ll_s = *((int*)ptr);
    // printf("c ll_s:%d\n", ll_s);
    ptr += sizeof(int);
    w->qkv_proj->q = (int8_t*)ptr;
    ptr += ll;
    w->qkv_proj->s = (float*)ptr;
    ptr += ll_s * sizeof(float);

    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    w->q_norm = (float*)ptr;
    ptr += ll * sizeof(float);

    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    w->k_norm = (float*)ptr;
    ptr += ll * sizeof(float);

    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    ll_s = *((int*)ptr);
    // printf("c ll_s:%d\n", ll_s);
    ptr += sizeof(int);
    w->out_proj->q = (int8_t*)ptr;
    ptr += ll;
    w->out_proj->s = (float*)ptr;
    ptr += ll_s * sizeof(float);

    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    w->ffn_norm = (float*)ptr;
    ptr += ll * sizeof(float);

    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    ll_s = *((int*)ptr);
    // printf("c ll_s:%d\n", ll_s);
    ptr += sizeof(int);
    w->proj_1->q = (int8_t*)ptr;
    ptr += ll;
    w->proj_1->s = (float*)ptr;
    ptr += ll_s * sizeof(float);

    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    ll_s = *((int*)ptr);
    // printf("c ll_s:%d\n", ll_s);
    ptr += sizeof(int);
    w->proj_2->q = (int8_t*)ptr;
    ptr += ll;
    w->proj_2->s = (float*)ptr;
    ptr += ll_s * sizeof(float);

    ll = *((int*)ptr);
    // printf("c ll:%d\n", ll);
    ptr += sizeof(int);
    w->norm = (float*)ptr;
    ptr += ll * sizeof(float);
    // printf("+++++++++++%d\n", w->qkv_proj->q[0]);
}

void openelm_build_from_checkpoint(OpenELM *model, char* checkpoint_path) {
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

    model->params_memory = (char*)malloc(model_size);
    fread(model->params_memory, sizeof(char), model_size, model_file);
    // for (int i = 0; i < 64; i++) {
    //     printf("weight: %f ", *(model->params_memory+i));
    // }
    // model->weights.token_embedding_table = model->params_memory;

    memory_map_weights(model->weights, &model->config, model->params_memory);
}

typedef struct {

} Context;

typedef struct {
    int batch;
    int length;
    int* data;
} Prompt; 

void read_prompt(Prompt *prompt, char* prompt_path) {
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

// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
void linear_forward(float* output, float* input, int8_t *weight_q, float* weight_s, int batch, int seq_len, int in_features, int out_features) {
    // printf("linear_forward batch:%d seq_len:%d in_features:%d out_features:%d\n", batch, seq_len, in_features, out_features);
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        for (int l = 0; l < seq_len; l++) {
            for(int out = 0; out < out_features; out++) {
                int offset_out = b * seq_len * out_features + l * out_features + out;
                int offset_bias = out;
                float value = 0.0f;

                for (int in = 0; in < in_features; in++) {
                    int offset_in = b * seq_len * in_features + l * in_features + in;
                    int offset_weight = out * in_features + in;                 
                    value += input[offset_in] * (weight_q[offset_weight]) * weight_s[out];

                }

                output[offset_out] = value;
            }
        }
    }
}

// https://arxiv.org/pdf/1910.07467
void rmsnorm_forward(float* output, float* input, float *weight, int batch, int seq_len, int dim) {
    // printf("rmsnorm_forward N:%d seq_len:%d dim:%d\n", batch, seq_len, dim);
    int b = 0;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        for (int l = 0; l < seq_len; l++) {
            
            int offset = b * seq_len * dim + l * dim;
            
            float ss = 0.0f;
            for (int d = 0; d < dim; d++) {
                ss += input[offset + d] * input[ offset + d];
            }
            ss /= dim;
            ss += 1e-6f;
            ss = 1.0f / sqrtf(ss);
            
            for (int d = 0; d < dim; d++) {
                 output[offset + d] = input[offset + d] * ss * weight[d];
            }
        }
    }
}

void rmsnorm_rope_forward(float* input, RunState *s, OpenELMWeights *w, OpenELMConfig *p, int layer_idx, int pos) {
    int batch = s->batch;
    int q_heads = p->num_query_heads[layer_idx];
    int k_heads = p->num_kv_heads[layer_idx];
    int v_heads = p->num_kv_heads[layer_idx];
    int head_dim = p->head_dim;
    int out_features = (q_heads + k_heads + v_heads) * head_dim;
    // printf("rmsnorm_forward N:%d layer_idx:%d pos:%d\n", batch, layer_idx, pos);
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        for(int out = 0; out < out_features; out++) {
            int offset_out = b * out_features + out;
            int offset_q = b * q_heads * head_dim;
            // int offset_k = layer_idx * s->max_seq_len * batch * s->max_kv_heads * head_dim 
            //              + pos * batch * s->max_kv_heads * head_dim
            //              + b * k_heads * head_dim;
            
            // int offset_k = b * s->max_seq_len * p->num_transformer_layers * s->max_kv_heads * head_dim 
            //              + pos * p->num_transformer_layers * s->max_kv_heads * head_dim 
            //              + layer_idx * s->max_kv_heads * head_dim 
            //              + k_heads * head_dim;

            // int offset_v = layer_idx * s->max_seq_len * batch * s->max_kv_heads * head_dim 
            //              + pos * batch * s->max_kv_heads * head_dim
            //              + b * v_heads * head_dim;
            int offset_v = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
                         + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
                         + pos * s->max_kv_heads * head_dim;
            int offset_bias = out;

            float value = 0.0f;
            for (int in = 0; in < p->model_dim; in++) {
                int offset_in = b * p->model_dim + in;
                int offset_weight = out * p->model_dim + in;      
                value += input[offset_in] * (float)((w->qkv_proj->q + s->qkv_proj_offset_q)[offset_weight]) * (w->qkv_proj->s + s->qkv_proj_offset_s)[out];
            }

            if (out < q_heads * head_dim) {
                s->q[offset_q + out] = value;
            } else if (out < (q_heads + k_heads) * head_dim) {
                int offset_k = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
                         + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
                         + pos * s->max_kv_heads * head_dim;
                // if (offset_k + out - q_heads * head_dim == 0) {
                //     printf("batch:%d, num_transformer_layers:%d, max_seq_len:%d, kv_dim:%d, offset_k:%d pos:%d value=%f\n", batch, p->num_transformer_layers, s->max_seq_len, s->max_kv_heads * head_dim, offset_k, pos, value);
                // }
                s->key_cache[offset_k + out - q_heads * head_dim] = value;
            } else if (out < (q_heads + k_heads + v_heads) * head_dim) {
                s->value_cache[offset_v + out - (q_heads + k_heads) * head_dim] = value;
            }
        }
    }
    
    // // printf qeury
    // if (layer_idx == 0) {
    // for (int b = 0; b < batch; b++) {
    //     printf("[");
    //     for (int h = 0; h < q_heads; h++) {
    //         printf("[");
    //         printf("[");
                    
    //         int offset = b * q_heads * head_dim + h * head_dim;
    //         for (int hd = 0; hd < head_dim; hd++) {     
    //             printf("%f,", s->q[offset + hd]);
    //         }
    //         printf("],\n");
    //         printf("],\n");
    //     }
    //     printf("],\n");
    // }
    // }

    // // printf key
    // if (layer_idx == 0) {
    // for (int b = 0; b < batch; b++) {
    //     printf("[");
    //     for (int h = 0; h < k_heads; h++) {
    //         printf("[");
    //         printf("[");
    //         int offset = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + pos * s->max_kv_heads * head_dim 
    //                      + h * head_dim;
    //         printf("offset=%d ", offset);
    //         for (int hd = 0; hd < head_dim; hd++) {
    //             printf("%f,", s->key_cache[offset + hd]);
    //         }
    //         printf("],\n");
    //         printf("],\n");
    //     }
    //     printf("],\n");
    // }
    // }

    // printf key
    // if (layer_idx == 0 && pos == 0) {
    // for (int b = 0; b < batch; b++) {
    //     printf("[");
    //     for (int h = 0; h < k_heads; h++) {
    //         printf("[");
    //         printf("[");
    //         int offset = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + pos * s->max_kv_heads * head_dim
    //                      + h * head_dim;
    //         printf("offset=%d ", offset);
    //         for (int hd = 0; hd < head_dim; hd++) {
    //             printf("%f,", s->key_cache[offset + hd]);
    //         }
    //         printf("],\n");
    //         printf("],\n");
    //     }
    //     printf("],\n");
    // }
    // }

    // // printf value
    // if (layer_idx == 0) {
    // for (int b = 0; b < batch; b++) {
    //     printf("[");
    //     for (int h = 0; h < v_heads; h++) {
    //         printf("[");
    //         printf("[");
    //         int offset = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + pos * s->max_kv_heads * head_dim
    //                      + h * head_dim;
    //         printf("offset=%d ", offset);
    //         for (int hd = 0; hd < head_dim; hd++) {
    //             printf("%f,", s->value_cache[offset + hd]);
    //         }
    //         printf("],\n");
    //         printf("],\n");
    //     }
    //     printf("],\n");
    // }
    // }

    s->qkv_proj_offset_q += out_features * p->model_dim;
    s->qkv_proj_offset_s += out_features;

    float rope_freq_constant = (float)p->rope_freq_constant;
    // printf("rope_freq_constant:%f\n", rope_freq_constant);
    // printf("rmsnorm_rope_forward N:%d seq_len:%d dim:%d\n", batch, seq_len, out_features);
    // int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        for (int h = 0; h < q_heads; h++) {
            int offset = b * q_heads * head_dim + h * head_dim;
            float ss = 0.0f;
            for (int hd = 0; hd < head_dim; hd++) {
                ss += s->q[offset + hd] * s->q[ offset + hd];
            }
            ss /= head_dim;
            ss += 1e-6f;
            ss = 1.0f / sqrtf(ss);
    
            for (int hd = 0; hd < head_dim; hd++) {
                s->q[offset + hd] = s->q[offset + hd] * ss * (w->q_norm + layer_idx * head_dim)[hd];
            }

            // https://arxiv.org/pdf/2104.09864

            for (int hd = 0; hd < head_dim / 2; hd++) {
                float v0 = s->q[offset + hd];
                float v1 = s->q[offset + hd + head_dim / 2];

                float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
                // printf("sl=%d %d=%f ", sl, hd, sl * freq);
                float cos_val = cosf(pos * freq);
                float sin_val = sinf(pos * freq);
                // printf("sl=%d %d=%f ", sl, hd, sin_val);
                s->q[offset + hd] = v0 * cos_val - v1 * sin_val;
                s->q[offset + head_dim / 2 + hd] = v1 * cos_val + v0 * sin_val;
                // s->x_qkv_proj[offset + hd + head_dim / 2] = v0 * sin_val + v1 * cos_val;
                // printf("batch=%d seq_len=%d heads=%d %d=%f %f v=%f %f cos_sin=%f %f\n", b, sl, h, hd, s->x_qkv_proj[offset + hd], s->x_qkv_proj[offset + head_dim / 2 + hd], 
                //        v0, v1, cos_val, sin_val);

                // printf("batch=%d seq_len=%d heads=%d %d=%f %f\n", b, sl, h, hd, s->x_qkv_proj[offset + hd], s->x_qkv_proj[offset + head_dim / 2 + hd]);
                // printf("batch=%d seq_len=%d heads=%d %d=%f %f v=%f %f cos_sin=%f %f\n", b, sl, h, hd, s->x_qkv_proj[offset + hd], s->x_qkv_proj[offset + head_dim / 2 + hd], v0, v1, cos_val, sin_val);
            }
        }
    }

    // // printf qeury
    // if (layer_idx == 0) {
    // for (int b = 0; b < batch; b++) {
    //     printf("[");
    //     for (int h = 0; h < q_heads; h++) {
    //         printf("[");
    //         printf("[");
                    
    //         int offset = b * q_heads * head_dim + h * head_dim;
    //         for (int hd = 0; hd < head_dim; hd++) {     
    //             printf("%f,", s->q[offset + hd]);
    //         }
    //         printf("],\n");
    //         printf("],\n");
    //     }
    //     printf("],\n");
    // }
    // }

    // int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        for (int h = 0; h < k_heads; h++) {
            int offset = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
                         + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
                         + pos * s->max_kv_heads * head_dim
                         + h * head_dim;
            float ss = 0.0f;
            for (int hd = 0; hd < head_dim; hd++) {
                ss += s->key_cache[offset + hd] * s->key_cache[ offset + hd];
            }
            ss /= head_dim;
            ss += 1e-6f;
            ss = 1.0f / sqrtf(ss);
            // https://arxiv.org/pdf/2104.09864

            for (int hd = 0; hd < head_dim; hd++) {
                s->key_cache[offset + hd] = s->key_cache[offset + hd] * ss * (w->k_norm  + layer_idx * head_dim)[hd];
            }

            for (int hd = 0; hd < head_dim / 2; hd++) {
                float v0 = s->key_cache[offset + hd];
                float v1 = s->key_cache[offset + hd + head_dim / 2];

                float freq = 1.0f / powf(rope_freq_constant, ((float)(2 * hd) / head_dim));
                // printf("sl=%d %d=%f ", sl, hd, sl * freq);
                float cos_val = cosf(pos * freq);
                float sin_val = sinf(pos * freq);
                // printf("sl=%d %d=%f ", sl, hd, sin_val);
                s->key_cache[offset + hd] = v0 * cos_val - v1 * sin_val;
                s->key_cache[offset + head_dim / 2 + hd] = v1 * cos_val + v0 * sin_val;
            }
        }
    }

    // if (layer_idx == 0) {
    // for (int b = 0; b < batch; b++) {
    //     printf("[");
    //     for (int h = 0; h < k_heads; h++) {
    //         printf("[");
    //         printf("[");
    //         int offset = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + 0 * s->max_kv_heads * head_dim
    //                      + h * head_dim;
    //         printf("offset=%d ", offset);
    //         for (int hd = 0; hd < head_dim; hd++) {
    //             printf("%f,", s->key_cache[offset + hd]);
    //         }
    //         printf("],\n");
    //         printf("],\n");
    //     }
    //     printf("],\n");
    // }
    // }

    // // printf key
    // if (layer_idx == 0 && pos == 0) {
    // for (int b = 0; b < batch; b++) {
    //     printf("[");
    //     for (int h = 0; h < k_heads; h++) {
    //         printf("[");
    //         printf("[");
    //         int offset = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + pos * s->max_kv_heads * head_dim
    //                      + h * head_dim;
    //         printf("offset=%d ", offset);
    //         for (int hd = 0; hd < head_dim; hd++) {
    //             printf("%f,", s->key_cache[offset + hd]);
    //         }
    //         printf("],\n");
    //         printf("],\n");
    //     }
    //     printf("],\n");
    // }
    // }

    // if (layer_idx == 0  && pos == 3) {
    // for (int b = 0; b < batch; b++) {
    //     printf("[");
    //     for (int h = 0; h < k_heads; h++) {
    //         printf("[");
    //         printf("[");
    //         int offset = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + 0 * s->max_kv_heads * head_dim
    //                      + h * head_dim;
    //         printf("offset=%d ", offset);
    //         for (int hd = 0; hd < head_dim; hd++) {
    //             printf("%f,", s->key_cache[offset + hd]);
    //         }
    //         printf("],\n");
    //         printf("],\n");
    //     }
    //     printf("],\n");
    // }
    // }

}

void group_attention_forward(float* output, RunState *s, OpenELMWeights *w, OpenELMConfig *p, int layer_idx, int pos) {
    int batch = s->batch;
    int q_heads = p->num_query_heads[layer_idx];
    int k_heads = p->num_kv_heads[layer_idx];
    int v_heads = p->num_kv_heads[layer_idx];
    // int num_heads = q_heads + k_heads + v_heads;
    // int model_dim = p->model_dim;
    int head_dim = p->head_dim;
    int num_groups = q_heads / k_heads;
    int out_features = (q_heads + k_heads + v_heads) * head_dim;
    
    // printf("group_attention_forward N:%d seq_len:%d head_dim:%d\n", s->batch, seq_len, head_dim);
    float min_dtype = -INFINITY;

    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        for (int h = 0; h < q_heads; h++) {
            int offset_att = b * s->max_q_heads * s->max_seq_len + h * s->max_seq_len;
            int offset_q = b * q_heads * head_dim + h * head_dim;

            for (int lk = 0; lk < pos + 1; lk++) {

                int offset_k = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
                         + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
                         + lk * s->max_kv_heads * head_dim
                         + (h / num_groups)  * head_dim;

                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += s->q[offset_q + i] * s->key_cache[offset_k + i];
                        // if (h == 0 && lk == 0) {
                        //     printf("offset_k:%d batch:%d, i:%d, q:%f, k:%f\n", offset_k, b, i, s->q[offset_q+i], s->key_cache[offset_k + i]);
                        // }
                }

                score /= sqrtf((float)head_dim);
                s->att[offset_att + lk] = score;
                // printf("%f ", score);
            }
            // printf("\n");
            float max_val = s->att[offset_att];
            for (int lk = 1; lk < pos + 1; lk++) { 
                if (s->att[offset_att + lk] > max_val) {
                    max_val = s->att[offset_att + lk];
                }
            }
            float ss = 0.0f;
            for (int lk = 0; lk < pos + 1; lk++) { 
                ss += expf(s->att[offset_att + lk] - max_val);
            }

            for (int lk = 0; lk < pos + 1; lk++) { 
                s->att[offset_att + lk] = expf(s->att[offset_att + lk] - max_val) / ss;
            }
                
            int offset_o = b * q_heads * head_dim + h * head_dim;
            for (int lv = 0; lv < head_dim; lv++){
                float sv = 0.0f;
                for (int k = 0; k < pos + 1; k++) { 
                    int offset_v = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
                         + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
                         + k * s->max_kv_heads * head_dim
                         + (h / num_groups) * head_dim;
                    sv += s->att[offset_att + k] * (s->value_cache[offset_v + lv]);
                }
                output[offset_o + lv] = sv;
            }
        }
    }
        // printf("[layer_idx=%d ", layer_idx);
        // for (int b = 0; b < batch; b++) {
        //     printf("[b=%d ", b);
        //     for (int h = 0; h < 1; h++) {
        //         printf("[h=%d ", h);
        //         for (int po = 0; po <= pos; po++) {
        //             int offset = b * s->max_seq_len * s->max_q_heads + h * s->max_seq_len + po;
        //             printf("%f, ", s->att[offset]);
        //         }
        //         printf("],\n");
        //     }
        //     printf("],\n");
        // }

    // for (int b = 0; b < batch; b++) {
    //     printf("[");
    //     for (int h = 0; h < v_heads; h++) {
    //         printf("[");
    //         printf("[");
    //         int offset = b * p->num_transformer_layers * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + layer_idx * s->max_seq_len * s->max_kv_heads * head_dim 
    //                      + pos * s->max_kv_heads * head_dim
    //                      + h * head_dim;
    //         printf("offset=%d ", offset);
    //         for (int hd = 0; hd < head_dim; hd++) {
    //             printf("%f,", s->value_cache[offset + hd]);
    //         }
    //         printf("],\n");
    //         printf("],\n");
    //     }
    //     printf("],\n");
    // }

        // for (int b = 0; b < batch; b++) {
        //     printf("[");
        //     for (int d = 0; d < q_heads * head_dim; d++) {
        //         int offset = b * q_heads * head_dim;
        //             printf("%f, ",output[offset + d]);
        //     }
        //     printf("],\n");
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

void glu_forward(float* output, float* input, RunState *s, OpenELMWeights *w, OpenELMConfig *p, int seq_len, int layer_idx) {
    
    int batch = s->batch;
    int model_dim = p->model_dim;
    float ffn_multiplier = p->ffn_multipliers[layer_idx];
    int intermediate_dim = (int)make_divisible(ffn_multiplier * p->model_dim, p->ffn_dim_divisor, -1);
    // printf("glu_forward batch:%d seq_len:%d model_dim:%d intermediate_dim:%d\n", s->batch, seq_len, p->model_dim, intermediate_dim);

    linear_forward(s->ihb, input, w->proj_1->q + s->proj_1_offset_q, w->proj_1->s + s->proj_1_offset_s, batch, seq_len, model_dim, 2 * intermediate_dim);
    s->proj_1_offset_q += 2 * intermediate_dim * model_dim;
    s->proj_1_offset_s += 2 * intermediate_dim;
    // // for (int i = batch * seq_len * model_dim - model_dim; i < batch * seq_len * model_dim; i++) {
    // for (int i = 0; i < batch * seq_len * 2 * intermediate_dim; i++) {
    //     printf("%d=%f ", i, s->hb[i]);
    // }

    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < s->batch; b++) {
        for (int sl = 0; sl < seq_len; sl++) {
            int offset_y1 = b * seq_len * 2 * intermediate_dim + sl * 2 * intermediate_dim;
            int offset_y2 = b * seq_len * 2 * intermediate_dim + sl * 2 * intermediate_dim + intermediate_dim;
            int offset_h = b * seq_len * intermediate_dim + sl * intermediate_dim;
            for (int d = 0; d < intermediate_dim; d++) {
                s->ihb[offset_y1 + d] = s->ihb[offset_y1 + d] / (1 + expf(-(s->ihb[offset_y1 + d])));
                s->ihb2[offset_h+ d] = s->ihb[offset_y1 + d] * s->ihb[offset_y2 + d];
            }
        }
    }

    // // for (int i = batch * seq_len * model_dim - model_dim; i < batch * seq_len * model_dim; i++) {
    // for (int i = 0; i < batch * seq_len * intermediate_dim; i++) {
    //     printf("%d=%f ", i, s->hb2[i]);
    // }
    
    linear_forward(output, s->ihb2, w->proj_2->q + s->proj_2_offset_q, w->proj_2->s + s->proj_2_offset_s, batch, seq_len, intermediate_dim, model_dim);
    s->proj_2_offset_q += intermediate_dim * model_dim;
    s->proj_2_offset_s += model_dim;
    // for (int i = 0; i < batch * seq_len * model_dim; i++) {
    //     printf("%d=%f ", i, output[i]);
    // }
    
}

void logits_forward(float* output, float* input, float *weight, float* bias, int batch, int seq_len, int in_features, int out_features) {
    // printf("logits_forward batch:%d seq_len:%d in_features:%d out_features:%d\n", batch, seq_len, in_features, out_features);
    int b;
    #pragma omp parallel for private(b)
    for (b = 0; b < batch; b++) {
        int l = seq_len - 1;
        // #pragma omp parallel for private(l)
        // for (l = 0; l < seq_len; l++) {
            for(int out = 0; out < out_features; out++) {
                int offset_out = b * out_features + out;
                int offset_bias = out;
                float value = 0.0f;
                for (int in = 0; in < in_features; in++) {
                    int offset_in = b * seq_len * in_features + l * in_features + in;
                    int offset_weight = out * in_features + in;
                    value += input[offset_in] * weight[offset_weight];

                }
                output[offset_out] = value;
                if (bias != NULL) {
                    output[offset_out] += bias[offset_bias];
                } 
            }
        // }
    }
}


void argmax_forward(int* output, float* input, int M, int N) {
    // printf("argmax_forward M:%d N:%d\n", M, N);
    int m = 0;
    #pragma omp parallel for private(m)
    for (m = 0; m < M; m++) {
        int v = 0;
        for (int n = 1; n < N; n++) {
           if (input[m * N + n] > input[m * N + v]) {
               v = n;
           }
        }
        output[m] = v;
    }
}

float* openelm_forward(Context *ctx, OpenELM* openelm, int *token, int batch, int pos) {
    OpenELMConfig *p = &openelm->config;
    OpenELMWeights *w = openelm->weights;
    RunState* s = &openelm->state;
    s->qkv_proj_offset_q = 0;
    s->qkv_proj_offset_s = 0;
    s->out_proj_offset_q = 0;
    s->out_proj_offset_s = 0;
    s->proj_1_offset_q = 0;
    s->proj_1_offset_s = 0;
    s->proj_2_offset_q = 0;
    s->proj_2_offset_s = 0;
    float *x = s->x;
    int model_dim = p->model_dim;
    int head_dim = p->head_dim;
    float rope_freq_constant = (float)p->rope_freq_constant;

    // printf("pos:%d, batch:%d, model_dim:%d \n", pos, batch, model_dim);
    for (int i = 0; i < batch; i++) {
        int offset_x = i * model_dim;
        float *content_row = w->token_embeddings + token[i] * model_dim;
        memcpy(x + offset_x, content_row, model_dim*sizeof(*x));
    }
    
    // printf("++++++++++++++++++++++ pos:%d\n", pos);
    // for (int i = 0; i < batch * model_dim; i++) {
    //     printf("%d=%f ", i, *(x + i));
    // }

    // printf("\n");

    // for(int l = 0; l < 1; l++) {
    for(int l = 0; l < p->num_transformer_layers; l++) {
        // printf("++++++++++++++++++++++ pos:%d\n", l);
        rmsnorm_forward(s->xb, s->x, w->attn_norm + l*model_dim, batch, 1, model_dim);
        // if (l == 0) {
        // for (int b = 0; b < batch; b++) {
        //     printf("[");
        //     for (int d = 0; d < model_dim; d++) {
        //         int offset = b * model_dim;
        //             printf("%f, ",s->xb[offset + d]);
        //     }
        //     printf("],\n");
        // }
        // }
        

        rmsnorm_rope_forward(s->xb, s, w, p, l, pos);
        // for (int i = batch * seq_len * out_features - out_features; i < batch * seq_len * out_features; i++) {
        // // for (int i = 0; i < out_features; i++) {
        //     printf("%d=%f ", i, s->x_qkv_proj[i]);
        // }
        int q_heads = p->num_query_heads[l];
        int k_heads = p->num_kv_heads[l];
        int v_heads = p->num_kv_heads[l];
        int out_features = (q_heads + k_heads + v_heads) * head_dim;

        group_attention_forward(s->xb, s, w, p, l, pos);
        // // for (int i = batch * seq_len * q_heads * head_dim - q_heads * head_dim; i < batch * seq_len * q_heads * head_dim; i++) {
        // if (l == 0) {
        // for (int i = 0; i < batch * q_heads * head_dim; i++) {
        //     printf("%d=%f ", i, s->xb[i]);
        // }
        // }
        // if (l == 5) {
        // for (int b = 0; b < batch; b++) {
        //     printf("[");
        //     for (int d = 0; d < q_heads * head_dim; d++) {
        //         int offset = b * q_heads * head_dim;
        //             printf("%f, ",s->xb[offset + d]);
        //     }
        //     printf("],\n");
        // }
        // }

        // if (l == 5) {
        // for (int b = 0; b < batch; b++) {
        //     printf("[");
        //     for (int h = 0; h < 1; h++) {
        //         printf("[");
        //         for (int p = 0; p <= pos; p++) {
        //             int offset = b * s->max_seq_len * s->max_q_heads + h * s->max_seq_len + p;
        //             printf("%f, ", s->att[offset]);
        //         }
        //         printf("],\n");
        //     }
        //     printf("],\n");
        // }
        // }


        // for (int i = 0; i < batch * q_heads *seq_len * seq_len; i++) {
        //     printf("%d=%f ", i, s->xb[i]);
        // }

        linear_forward(s->xb2, s->xb, w->out_proj->q + s->out_proj_offset_q, w->out_proj->s + s->out_proj_offset_s, batch, 1, q_heads * head_dim, model_dim);
        s->out_proj_offset_q += q_heads * head_dim * model_dim;
        s->out_proj_offset_s += model_dim;

        // if (l == 0) {
        // for (int b = 0; b < batch; b++) {
        //     printf("[");
        //     for (int d = 0; d < model_dim; d++) {
        //         int offset = b * model_dim;
        //             printf("%f, ",s->xb2[offset + d]);
        //     }
        //     printf("],\n");
        // }
        // }

        for (int i = 0; i < batch * model_dim; i++) {
            s->x[i] += s->xb2[i];
        }

        // if (l == 0) {
        // for (int b = 0; b < batch; b++) {
        //     printf("[");
        //     for (int d = 0; d < model_dim; d++) {
        //         int offset = b * model_dim;
        //             printf("%f, ",s->x[offset + d]);
        //     }
        //     printf("],\n");
        // }
        // }

        rmsnorm_forward(s->xb, s->x, w->ffn_norm + l*model_dim, batch, 1, model_dim);
        // if (l == 5) {
        // for (int b = 0; b < batch; b++) {
        //     printf("[");
        //     for (int d = 0; d < model_dim; d++) {
        //         int offset = b * model_dim;
        //             printf("%f, ",s->xb[offset + d]);
        //     }
        //     printf("],\n");
        // }
        // }

        glu_forward(s->hb, s->xb, s, w, p, 1, l);
        
        for (int i = 0; i < batch * model_dim; i++) {
            s->x[i] += s->hb[i];
        }
        
        // if (l == 5) {
        // for (int b = 0; b < batch; b++) {
        //     printf("[");
        //     for (int d = 0; d < model_dim; d++) {
        //         int offset = b * model_dim;
        //             printf("%f, ",s->x[offset + d]);
        //     }
        //     printf("],\n");
        // }
        // }
    }

    rmsnorm_forward(s->x, s->x, w->norm, batch, 1, model_dim);

        // for (int b = 0; b < batch; b++) {
        //     printf("[");
        //     for (int d = 0; d < model_dim; d++) {
        //         int offset = b * model_dim;
        //             printf("%f, ",s->x[offset + d]);
        //     }
        //     printf("],\n");
        // }

    logits_forward(s->logits, s->x, w->token_embeddings, NULL, batch, 1, model_dim, p->vocab_size);

    // for (int i = 0; i < batch * p->vocab_size; i++) {
    //     printf("%d=%f ", i, s->logits[i]);
    // }

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

    int num_prompt_tokens = prompt->length;
    int* prompt_tokens = prompt->data;

    int pos = 0;
    
    for (int i = 0; i < prompt->batch; i++) {
        int offset = i * prompt->length + pos;
        s->token[i] = prompt->data[offset];
        // printf("token %d=%d\n", i, s->token[i]);
    }

    long start = time_in_ms();
    while (pos < steps) {
        float *logits = openelm_forward(ctx, openelm, s->token, prompt->batch, pos);

        argmax_forward(s->next, s->logits, s->batch, openelm->config.vocab_size);

        for (int i = 0; i < s->batch; i++) {
            printf("%d=%d ", i, s->next[i]);
        }
        // if (pos < num_prompt_tokens - 1) {
        //     for (int i = 0; i < prompt->batch; i++) {
        //         next[i] = *(prompt->data + i * prompt->length + pos + 1);
        //     }
        // }
        pos++;

        if (pos < prompt->length) {
            for (int i = 0; i < s->batch; i++) {
                int offset = i * prompt->length + pos ;
                s->next[i] = prompt->data[offset];
            }
        }

        // for (int i = 0; i < s->batch; i++) {
        //     printf("%d=%d ", i, s->next[i]);
        // }
        printf("\n");

        for (int i = 0; i < s->batch; i++) {
            s->token[i] = s->next[i];
        }

        if (pos == prompt->length) {
            break;
        }
        

        // for (int i = 0; i < prompt->batch; i++) {
        //     token[i] = next[i];
        // }
        // if (start == 0) { start = time_in_ms(); }
    }
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);

}

OpenELM *c_get_model() {
    static OpenELM *model = NULL;
    if (model == NULL) {
        model = (OpenELM*)malloc(sizeof(OpenELM));
        OpenELMWeights *weights = (OpenELMWeights*)malloc(sizeof(OpenELMWeights));
        model->weights = weights;
        QuantizedTensor *qkv_proj = (QuantizedTensor*)malloc(sizeof(QuantizedTensor));
        model->weights->qkv_proj = qkv_proj;
        QuantizedTensor *out_proj = (QuantizedTensor*)malloc(sizeof(QuantizedTensor));
        model->weights->out_proj = out_proj;
        QuantizedTensor *proj_1 = (QuantizedTensor*)malloc(sizeof(QuantizedTensor));
        model->weights->proj_1 = proj_1;
        QuantizedTensor *proj_2 = (QuantizedTensor*)malloc(sizeof(QuantizedTensor));
        model->weights->proj_2 = proj_2;
    }

    return model;
}

void c_init(int batch, int max_seq_len) {
    OpenELM *model = c_get_model();

    openelm_build_from_checkpoint(model, "openelm_270M_q80.bin");
    model->state.batch = batch;
    model->state.max_seq_len = max_seq_len;
    malloc_run_state(&model->state, &model->config);
}

// void get_mod
int* c_openelm_forward(int batch, int seq_len, int *data, int pos) {
    // printf("c_openelm_forward batch:%d, seq_len:%d, pos:%d\n", batch, seq_len, pos);
    OpenELM *model = c_get_model();
    RunState *s = &model->state;
    
    int* prompt_tokens = data;
    int start = 0;
    for (int i = 0; i < batch; i++) {
        s->token[i] = data[i];
        // printf("%d ", s->token[i]);
    }
    
    Context ctx;
    openelm_forward(&ctx, model, s->token, batch, pos);
    argmax_forward(s->sample, s->logits, s->batch, model->config.vocab_size);
    // for (int i = 0; i < s->batch; i++) {
    //     printf("%d=%d ", i, s->sample[i]);
    // }
    // printf("\n");
    return s->sample;
}

void c_generate(int batch, int seq_len, int *data, int steps) {
    OpenELM *model = c_get_model();
    Context ctx;
    Prompt prompt;
    prompt.batch = batch;
    prompt.length = seq_len;
    prompt.data = data;
    generate(&ctx, model, &prompt, steps);
    printf("hello openelm\n");
}

void c_chat () {

}


int start() {
    Prompt prompt;
    read_prompt(&prompt, "openelm_prompt.bin");

    OpenELM *model = (OpenELM*)malloc(sizeof(OpenELM));
    OpenELMWeights *weights = (OpenELMWeights*)malloc(sizeof(OpenELMWeights));
    model->weights = weights;
    QuantizedTensor qkv_proj;
    model->weights->qkv_proj = &qkv_proj;
    QuantizedTensor out_proj;
    model->weights->out_proj = &out_proj;
    QuantizedTensor proj_1;
    model->weights->proj_1 = &proj_1;
    QuantizedTensor proj_2;
    model->weights->proj_2 = &proj_2;
    OpenELMConfig config;

    openelm_build_from_checkpoint(model, "openelm_270M_q80.bin");

    model->state.batch = prompt.batch;
    model->state.max_seq_len = 256;
    malloc_run_state(&model->state, &model->config);
    
    Context ctx;
    generate(&ctx, model, &prompt, 256);
    printf("hello openelm\n");
}

int main(int argc, char** argv) {
    start();
}