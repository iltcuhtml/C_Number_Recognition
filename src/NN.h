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

float rand_float(void);

typedef struct
{
    size_t rows;
    size_t cols;

    float *es;
} Mat;

#define MAT_AT(m, row, col) (m).es[(row) * (m).cols + (col)]

Mat mat_alloc(size_t rows, size_t cols);

void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float x);

void mat_print(Mat m);
void mat_sum(Mat dst, Mat m);
void mat_dot(Mat dst, Mat m, Mat g);

#endif // NN_H

#ifdef NN_IMPLEMENTATION
#define NN_IMPLEMENTATION

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;

    m.rows = rows;
    m.cols = cols;
    m.es = (float*) malloc(sizeof(*m.es) * rows * cols);

    NN_ASSERT(m.es != NULL);

    return m;
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i)
        for (size_t ii = 0; ii < m.cols; ++ii)
        {
            MAT_AT(m, i, ii) = rand_float() * (high - low) + low;
        }
}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i)
        for (size_t ii = 0; ii < m.cols; ++ii)
        {
            MAT_AT(m, i, ii) = x;
        }
}

void mat_print(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t ii = 0; ii < m.cols; ++ii)
        {
            printf("%f ", MAT_AT(m, i, ii));
        }

        printf("\n");
    }
}

void mat_sum(Mat dst, Mat m)
{
    NN_ASSERT(dst.rows == m.rows);
    NN_ASSERT(dst.cols == m.cols);

    for (size_t i = 0; i < dst.rows; ++i)
        for (size_t ii = 0; ii < dst.cols; ++ii)
        {
            MAT_AT(dst, i, ii) += MAT_AT(m, i, ii);
        }
}

void mat_dot(Mat dst, Mat m, Mat g)
{
    NN_ASSERT(m.cols == g.rows);
    NN_ASSERT(dst.rows == m.rows);
    NN_ASSERT(dst.cols == g.cols);

    for (size_t i = 0; i < dst.rows; ++i)
        for (size_t ii = 0; ii < dst.cols; ++ii)
        {
            MAT_AT(dst, i, ii) = 0;

            for (size_t iii = 0; iii < m.cols; ++iii)
            {
                MAT_AT(dst, i, ii) += MAT_AT(m, i, iii) * MAT_AT(g, iii, ii);
            }
        }
}

#endif // NN_IMPLEMENTATION