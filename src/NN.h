#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

float sigmoidf(float x);

float rand_float(void);

typedef struct
{
    size_t rows;
    size_t cols;

    size_t stride;

    float *es;
} Mat;

#define MAT_AT(m, row, col) (m).es[(row) * (m).stride + (col)]

Mat mat_alloc(size_t rows, size_t cols);

void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float x);

void mat_print(Mat m, const char *name);
#define MAT_PRINT(m) mat_print(m, #m)

Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);

void mat_sum(Mat dst, Mat m);
void mat_dot(Mat dst, Mat m, Mat g);

void mat_sig(Mat m);

#endif // NN_H

#ifdef NN_IMPLEMENTATION
#define NN_IMPLEMENTATION

float sigmoidf(float x)
{
    return 1.0F / (1.0F + expf(-x));
}

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;

    m.rows = rows;
    m.cols = cols;

    m.stride = cols;

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

void mat_print(Mat m, const char *name)
{
    printf("\n%s = [\n", name);
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t ii = 0; ii < m.cols; ++ii)
        {
            printf("    %f", MAT_AT(m, i, ii));
        }

        printf("\n");
    }
    printf("]\n");
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat) {
        .rows = 1, 
        .cols = m.cols, 

        .stride = m.stride, 

        .es = &MAT_AT(m, row, 0)
    };
}

void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; ++i)
        for (size_t ii = 0; ii < dst.cols; ++ii)
        {
            MAT_AT(dst, i, ii) = MAT_AT(src, i, ii);
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

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i)
        for (size_t ii = 0; ii < m.cols; ++ii)
        {
            MAT_AT(m, i, ii) = sigmoidf(MAT_AT(m, i, ii));
        }
}

#endif // NN_IMPLEMENTATION