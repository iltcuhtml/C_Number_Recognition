#include <time.h>

#define NN_IMPLEMENTATION
#include "NN.h"

#define TRAIN_COUNT (size_t) 1E6

#define RATE (float) 1E-1

float AND_gate[] = {
    0, 0, 0,
    0, 1, 0,
    1, 0, 0,
    1, 1, 1,
};

float OR_gate[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

float NAND_gate[] = {
    0, 0, 1,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float XOR_gate[] = {
    0, 0, 0, 
    0, 1, 1, 
    1, 0, 1, 
    1, 1, 0, 
};

// td - train data
float *td = XOR_gate;

int main(void)
{
    srand(time(0));

    size_t stride = 3;
    size_t n = 4; // ARRAY_LEN(td) / stride;
    
    // ti - train input
    Mat ti = {
        .rows = n, 
        .cols = 2, 

        .stride = stride, 

        .es = td
    };

    // to - train output
    Mat to = {
        .rows = n, 
        .cols = 1, 

        .stride = stride, 

        .es = td + 2
    };

    size_t arch[] = { 2, 3, 1 };
    NN nn  = NN_alloc(arch, ARRAY_LEN(arch));
    NN gnn = NN_alloc(arch, ARRAY_LEN(arch));

    NN_rand(nn, 0, 1);

    for (size_t i = 0; i < TRAIN_COUNT; ++i)
    {
        NN_backprop(nn, gnn, ti, to);
        NN_learn(nn, gnn, RATE);
    }

    NN_PRINT(nn);

    printf("\ncost = %f\n", NN_cost(nn, ti, to));

    for (size_t i = 0; i < 2; ++i)
        for (size_t ii = 0; ii < 2; ++ii)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = ii;

            NN_forward(nn);

            printf("\n%zu ^ %zu = %f", i, ii, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    
    NN_free(nn);
    NN_free(gnn);

    return 0;
}