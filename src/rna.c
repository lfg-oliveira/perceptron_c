#include "rna.h"

#define MAX_EPOCHS 15
#define NUM_FEATURES 4
#define LEARNING_RATE 0.4
#define TRAIN_SIZE 6
#define TEST_SIZE 2

/*
** Perceptron
*/
rna_network* rna_init_network(int num_features, float learning_rate) {
    #ifndef SRAND_INIT
    #define SRAND_INIT
    srand(0xcafe);
    #endif
    rna_network *n =(rna_network*) calloc(1, sizeof(rna_network));
    n->weights = (float *) calloc(num_features, sizeof(float));
    n->weight_length = num_features;
    n->learning_rate = learning_rate;
    for(int i = 0; i < num_features ; i++) {
        n->weights[i] = ((float)rand()-(float)RAND_MAX/2)/(float)RAND_MAX;
        printf("Weight %d: %f ", i, n->weights[i]);
    }

    n->bias = 0.1*(rand()/(float)RAND_MAX - 0.5);

    return n;
}

float rna_sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

float rna_forward(rna_network* nw,  float *input) {
    if( !nw->weights ) {
        printf("Failed initing network");
        exit(1);
    }

    float acc = 0;
    // u = \sum{i=1,p} x_i*w_i - \theta
    // onde \theta é o bias
    for(int i = 0; i < nw->weight_length; i++) {
        acc += nw->weights[i] * input[i];
    }

    acc -= nw->bias;

    return acc;
}

float rna_backward(rna_network* nw, float *x, float pred, float val) {
    float error = val - pred;
    if(error == 0) return 0;
    // Wij(n)= Wij(n-1) + \n * x_i(Y_i - y)
    // onde \n é a taxa de aprendizagem
    // Y são os valores alvos
    // y é a saída da rede para x_i
    for(int i = 0; i< nw->weight_length; i++) {
        nw->weights[i] += x[i] * nw->learning_rate * error;
    }
    nw->bias += nw->learning_rate * -1 * error;

    return pred-val;
}

float rna_normalize(float val) {
    return val/fabsf(val);
}

void rna_destroy_nw(rna_network *nw) {
    free(nw->weights);
    free(nw);
}


int main() {
    rna_network* nw = rna_init_network(NUM_FEATURES, LEARNING_RATE);

    // mapa de valores: Sim/Grande/Saudável -> 1
    // Não/pequena/doente -> -1

    float input[TRAIN_SIZE][NUM_FEATURES] = {
        {1,-1, -1, 1},
        {-1, -1, 1, -1},
        {1, 1, -1, 1},
        {1, -1, -1, 1},
        {1, -1, -1, 1},
        {-1, -1, 1, 1}
    };
    float test[2][NUM_FEATURES] = {
        {-1,-1,-1,1},
        {1,1,1,1}
    };

    float output[TRAIN_SIZE] = {
        1,
        -1,
        1,
        -1,
        1,
        -1
    };
    for(int i = 0; i<MAX_EPOCHS; i++) {
        double error = 0;
        printf("\nErro: %f\n-----------------------\nIniciando nova época: Época %d\n", error, i+1);
        puts("\n");
        for(int j = 0; j < TRAIN_SIZE; j++) {
            float pred = rna_forward(nw, input[j]);
            printf("\nPredicao: %f | Alvo: %f", pred, output[j]);
            error += rna_normalize(pred) - output[j];
            (fabs(pred - output[j]) < 1e-4)? :
            rna_backward(nw, input[j], pred, output[j]);
        }
        if(fabs(error) < 1e-8) break;
        error /= TRAIN_SIZE;
    }

    puts("\n");
    for(int i = 0; i< TEST_SIZE; i++) {
       printf("\nTeste %d: %f", i, rna_forward(nw, test[i]));
    }
    rna_destroy_nw(nw);
}
