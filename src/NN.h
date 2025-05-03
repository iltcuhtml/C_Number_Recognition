#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <math.h>

#ifndef NN_EPS
#define NN_EPS 1E-1
#endif // NN_EPS

#ifndef NN_RATE
#define NN_RATE 1E-1
#endif // NN_RATE

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

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

Mat Mat_alloc(size_t rows, size_t cols);

void Mat_rand(Mat m, float low, float high);
void Mat_fill(Mat m, float x);

void Mat_print(Mat m, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)

Mat Mat_row(Mat m, size_t row);
void Mat_copy(Mat dst, Mat src);

void Mat_sum(Mat dst, Mat m);
void Mat_dot(Mat dst, Mat m1, Mat m2);

void Mat_sig(Mat m);

typedef struct
{
    size_t count;

    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count + 1
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN NN_alloc(size_t *arch, size_t arch_count);

void NN_rand(NN nn, float low, float high);

void NN_print(NN nn, const char *name);
#define NN_PRINT(nn) NN_print(nn, #nn)

void NN_forward(NN nn);

// tdi = Train Data Input | tdo = Train Data Output
float NN_cost(NN nn, Mat tdi, Mat tdo);

// dnn = diff_nn | tdi = Train Data Input | tdo = Train Data Output
void NN_finite_diff(NN nn, NN dnn, Mat tdi, Mat tdo);

// dnn = diff_nn
void NN_train(NN nn, NN dnn);

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

Mat Mat_alloc(size_t rows, size_t cols)
{
    Mat m;

    m.rows = rows;
    m.cols = cols;

    m.stride = cols;

    m.es = (float*) malloc(sizeof(*m.es) * rows * cols);

    NN_ASSERT(m.es != NULL);

    return m;
}

void Mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i)
        for (size_t ii = 0; ii < m.cols; ++ii)
        {
            MAT_AT(m, i, ii) = rand_float() * (high - low) + low;
        }
}

void Mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i)
        for (size_t ii = 0; ii < m.cols; ++ii)
        {
            MAT_AT(m, i, ii) = x;
        }
}

void Mat_print(Mat m, const char *name, size_t padding)
{
    printf("\n%*s%s = [\n", (int) padding, "", name);

    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t ii = 0; ii < m.cols; ++ii)
        {
            printf("%*s    %f", (int) padding, "", MAT_AT(m, i, ii));
        }

        printf("\n");
    }

    printf("%*s]\n", (int) padding, "");
}

Mat Mat_row(Mat m, size_t row)
{
    return (Mat) {
        .rows = 1, 
        .cols = m.cols, 

        .stride = m.stride, 

        .es = &MAT_AT(m, row, 0)
    };
}

void Mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; ++i)
        for (size_t ii = 0; ii < dst.cols; ++ii)
        {
            MAT_AT(dst, i, ii) = MAT_AT(src, i, ii);
        }
}

void Mat_sum(Mat dst, Mat m)
{
    NN_ASSERT(dst.rows == m.rows);
    NN_ASSERT(dst.cols == m.cols);

    for (size_t i = 0; i < dst.rows; ++i)
        for (size_t ii = 0; ii < dst.cols; ++ii)
        {
            MAT_AT(dst, i, ii) += MAT_AT(m, i, ii);
        }
}

void Mat_dot(Mat dst, Mat m1, Mat m2)
{
    NN_ASSERT(m1.cols == m2.rows);
    NN_ASSERT(dst.rows == m1.rows);
    NN_ASSERT(dst.cols == m2.cols);

    for (size_t i = 0; i < dst.rows; ++i)
        for (size_t ii = 0; ii < dst.cols; ++ii)
        {
            MAT_AT(dst, i, ii) = 0;

            for (size_t iii = 0; iii < m1.cols; ++iii)
            {
                MAT_AT(dst, i, ii) += MAT_AT(m1, i, iii) * MAT_AT(m2, iii, ii);
            }
        }
}

void Mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i)
        for (size_t ii = 0; ii < m.cols; ++ii)
        {
            MAT_AT(m, i, ii) = sigmoidf(MAT_AT(m, i, ii));
        }
}

NN NN_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);

    NN nn;

    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL);

    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL);
    
    nn.as = NN_MALLOC(sizeof(*nn.as) * nn.count);
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = Mat_alloc(1, arch[0]);

    for (size_t i = 0; i < nn.count; ++i)
    {
        nn.ws[i] = Mat_alloc(nn.as[i].cols, arch[i + 1]);
        nn.bs[i] = Mat_alloc(1, arch[i + 1]);
        nn.as[i + 1] = Mat_alloc(1, arch[i + 1]);
    }

    return nn;
}

void NN_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        Mat_rand(nn.ws[i], low, high);
        Mat_rand(nn.bs[i], low, high);
    }
}

void NN_print(NN nn, const char *name)
{
    char buf[256];

    printf("\n%s = [\n", name);
    
    for (size_t i = 0; i < nn.count; ++i)
    {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        Mat_print(nn.ws[i], buf, 4);
        
        snprintf(buf, sizeof(buf), "bs%zu", i);
        Mat_print(nn.bs[i], buf, 4);
    }

    printf("]\n");
}

void NN_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        Mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        Mat_sum(nn.as[i + 1], nn.bs[i]);
        Mat_sig(nn.as[i + 1]);
    }
}

float NN_cost(NN nn, Mat tdi, Mat tdo)
{
    assert(tdi.rows == tdo.rows);
    assert(tdo.cols == NN_OUTPUT(nn).cols);
    
    float c = 0.0F;

    for (size_t i = 0; i < tdi.rows; ++i)
    {
        Mat x = Mat_row(tdi, i);
        Mat y = Mat_row(tdo, i);
        
        Mat_copy(NN_INPUT(nn), x);
        NN_forward(nn);

        for (size_t ii = 0; ii < tdo.cols; ++ii)
        {
            float d = MAT_AT(NN_OUTPUT(nn), 0, ii) - MAT_AT(y, 0, ii);

            c += d * d;
        }
    }

    return c / tdi.rows;
}

void NN_finite_diff(NN nn, NN dnn, Mat tdi, Mat tdo)
{
    float saved;

    float c = NN_cost(nn, tdi, tdo);

    for (size_t i = 0; i < nn.count; ++i)
    {
        for (size_t ii = 0; ii < nn.ws[i].rows; ++ii)
            for (size_t iii = 0; iii < nn.ws[i].cols; ++iii)
            {
                saved = MAT_AT(nn.ws[i], ii, iii);

                MAT_AT(nn.ws[i], ii, iii) += NN_EPS;
                MAT_AT(dnn.ws[i], ii, iii) = (NN_cost(nn, tdi, tdo) - c) / NN_EPS;
                MAT_AT(nn.ws[i], ii, iii) = saved;
            }

        for (size_t ii = 0; ii < nn.bs[i].rows; ++ii)
            for (size_t iii = 0; iii < nn.bs[i].cols; ++iii)
            {
                saved = MAT_AT(nn.bs[i], ii, iii);
    
                MAT_AT(nn.bs[i], ii, iii) += NN_EPS;
                MAT_AT(dnn.bs[i], ii, iii) = (NN_cost(nn, tdi, tdo) - c) / NN_EPS;
                MAT_AT(nn.bs[i], ii, iii) = saved;
            }
    }
}

void NN_train(NN nn, NN dnn)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        for (size_t ii = 0; ii < nn.ws[i].rows; ++ii)
            for (size_t iii = 0; iii < nn.ws[i].cols; ++iii)
            {
                MAT_AT(nn.ws[i], ii, iii) -= MAT_AT(dnn.ws[i], ii, iii) * NN_RATE;
            }

        for (size_t ii = 0; ii < nn.bs[i].rows; ++ii)
            for (size_t iii = 0; iii < nn.bs[i].cols; ++iii)
            {    
                MAT_AT(nn.bs[i], ii, iii) -= MAT_AT(dnn.bs[i], ii, iii) * NN_RATE;
            }
    }
}

#endif // NN_IMPLEMENTATION