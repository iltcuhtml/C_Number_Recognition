#include <time.h>

#define NN_IMPLEMENTATION
#include "NN.h"

#define TRAIN_COUNT (size_t) 1E4

#define RATE (float) 1E-0

#define BITS (1 << 2)

int main(void)
{
    size_t n = (1 << BITS);
    size_t rows = n * n;

    Mat ti = Mat_alloc(rows, BITS * 2);
    Mat to = Mat_alloc(rows, BITS + 1);

    for (size_t i = 0; i < ti.rows; ++i)
    {
        size_t x = i / n;
        size_t y = i % n;

        size_t z = x + y;

        for (size_t ii = 0; ii < BITS; ++ii)
        {
            MAT_AT(ti, i, ii)        = (x >> ii) & 1;
            MAT_AT(ti, i, ii + BITS) = (y >> ii) & 1;
            MAT_AT(to, i, ii) = (z >> ii) & 1;
        }

        MAT_AT(to, i, BITS) = z >= n;
    }

    size_t arch[] = { BITS * 2, BITS * 4, BITS + 1};
    NN nn  = NN_alloc(arch, ARRAY_LEN(arch));
    NN gnn = NN_alloc(arch, ARRAY_LEN(arch));

    NN_rand(nn, 0, 1);

    for (size_t i = 0; i < TRAIN_COUNT; ++i)
    {
        NN_backprop(nn, gnn, ti, to);
        NN_train(nn, gnn, RATE);
    }
    
    printf("cost = %f\n\n", NN_cost(nn, ti, to));

    size_t fails = 0;

    for (size_t x = 0; x < n; ++x)
        for (size_t y = 0; y < n; ++y)
        {
            size_t z = x + y;

            for (size_t ii = 0; ii < BITS; ++ii)
            {
                MAT_AT(NN_INPUT(nn), 0, ii)        = (x >> ii) & 1;
                MAT_AT(NN_INPUT(nn), 0, ii + BITS) = (y >> ii) & 1;
            }
            
            NN_forward(nn);
            
            if (MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5F)
            {
                if (z < n)
                {
                    printf("%zu + %zu = (OVERFLOW <> %zu)\n", x, y, z);

                    fails += 1;
                }
            }
            else
            {
                size_t a = 0;

                for(size_t ii = 0; ii < BITS; ++ii)
                {
                    size_t bit = MAT_AT(NN_OUTPUT(nn), 0, ii) > 0.5F;

                    a |= bit << ii;
                }

                if (z != a)
                {
                    printf("%zu + %zu = (%zu <> %zu)\n", x, y, z, a);

                    fails += 1;
                }
            }
        }

    if (fails == 0)
    {
        printf("OK");
    }

    Mat_free(ti);
    Mat_free(to);

    NN_free(nn);
    NN_free(gnn);

    return 0;
}