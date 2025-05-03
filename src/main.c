#include <time.h>

#define NN_IMPLEMENTATION

#define NN_EPS  (float) 1E-4
#define NN_RATE (float) 1E-2

#include "NN.h"

#define TRAIN_COUNT (size_t) 1E6

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

float XOR_gate[] = {
    0, 0, 0, 
    0, 1, 1, 
    1, 0, 1, 
    1, 1, 0, 
};

float *td = XOR_gate;

int main(void) // int argc, char* argv[]
{
    srand(time(0));

    size_t stride = 3;
    size_t n = 4;// ARRAY_LEN(td) / stride;
    
    Mat tdi = {
        .rows = n, 
        .cols = 2, 

        .stride = stride, 

        .es = td
    };

    Mat tdo = {
        .rows = n, 
        .cols = 1, 

        .stride = stride, 

        .es = td + 2
    };

    size_t arch[] = { 2, 3, 1 };
    NN nn = NN_alloc(arch, ARRAY_LEN(arch));
    NN dnn  = NN_alloc(arch, ARRAY_LEN(arch));

    NN_rand(nn, 0, 1);

    for (size_t i = 0; i < TRAIN_COUNT; ++i)
    {
        NN_finite_diff(nn, dnn, tdi, tdo);
        NN_train(nn, dnn);
    }

    NN_PRINT(nn);

    printf("\ncost = %f\n", NN_cost(nn, tdi, tdo));

    for (size_t i = 0; i < 2; ++i)
        for (size_t ii = 0; ii < 2; ++ii)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = ii;

            NN_forward(nn);

            printf("\n%zu ^ %zu = %f", i, ii, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }

    Mat_free(tdi);
    Mat_free(tdo);
    
    NN_free(nn);
    NN_free(dnn);

    return 0;
}