#define NN_IMPLEMENTATION
#include "NN.h"

#define BITS 4

int main(void)
{
    size_t n = (1 << BITS);
    size_t rows = n * n;

    Mat t  = Mat_alloc(rows, (BITS * 2) + (BITS + 1));

    Mat ti = {
        .rows = t.rows, 
        .cols = BITS * 2, 

        .stride = t.stride, 

        .es = &MAT_AT(t, 0, 0)
    };

    Mat to = {
        .rows = t.rows, 
        .cols = BITS + 1, 

        .stride = t.stride, 

        .es = &MAT_AT(t, 0, BITS * 2)
    };

    for (size_t i = 0; i < ti.rows; i++)
    {
        size_t x = i / n;
        size_t y = i % n;

        size_t z = x + y;

        for (size_t ii = 0; ii < BITS; ii++)
        {
            MAT_AT(ti, i, ii)        = (x >> ii) & 1;
            MAT_AT(ti, i, ii + BITS) = (y >> ii) & 1;
            MAT_AT(to, i, ii) = (z >> ii) & 1;
        }

        MAT_AT(to, i, BITS) = z >= n;
    }

    const char *out_file_path = "adder.mat";
    FILE *out = fopen(out_file_path, "wb");

    if (out == NULL)
    {
        fprintf(stderr, "ERROR: Could not open file %s\n", out_file_path);

        return EXIT_FAILURE;
    }

    Mat_save(out, t);

    fclose(out);

    printf("Generated %s\n", out_file_path);

    Mat_free(t);

    return EXIT_SUCCESS;
}