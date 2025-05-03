#ifndef NN_H
#define NN_H

#include <stdio.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct
{
    size_t rows;
    size_t cols;

    float *es;
} Mat;

#define MAT_AT(m, row, col) (m).es[(row) * (m).cols + (col)]

Mat mat_alloc(size_t rows, size_t cols);

void mat_dot(Mat dst, Mat m, Mat g);
void mat_sum(Mat dst, Mat m);
void mat_print(Mat m);

#endif // NN_H

#ifndef NN_IMPLEMENTATION
#define NN_IMPLEMENTATION

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;

    m.rows = rows;
    m.cols = cols;
    m.es = (float*) malloc(sizeof(*m.es) * rows * cols);

    assert(m.es != NULL);

    return m;
}

void mat_dot(Mat dst, Mat m, Mat g)
{
    (void) dst;
    (void) m;
    (void) g;
}

void mat_sum(Mat dst, Mat m)
{
    (void) dst;
    (void) m;
}

void mat_print(Mat m)
{
    for (size_t row = 0; row < m.rows; ++row)
    {
        for (size_t col = 0; col < m.cols; ++col)
        {
            printf("%f ", MAT_AT(m, row, col));
        }

        printf("\n");
    }
}

#endif // NN_IMPLEMENTATION