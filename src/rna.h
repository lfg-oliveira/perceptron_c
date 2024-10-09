#ifndef RNA_H_
#define RNA_H_
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

typedef struct network {
    float *weights;
    int weight_length;
    float learning_rate;
    float bias;
} rna_network;

rna_network* rna_init_network(int, float);
float rna_forward(rna_network*,  float *input);
float rna_backward(rna_network*, float *, float, float);
void rna_destroy_nw(rna_network*);
float rna_normalize(float val);

#endif // RNA_H_
