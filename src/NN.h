#pragma once

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

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

void Mat_save(FILE *out, Mat m);
Mat Mat_load(FILE *in);

void Mat_rand(Mat m, float low, float high);
void Mat_fill(Mat m, float x);

void Mat_print(Mat m, const char *name, size_t padding);
#define MAT_PRINT(m) Mat_print(m, #m, 0)

Mat Mat_row(Mat m, size_t row);
void Mat_copy(Mat dst, Mat src);

void Mat_sum(Mat dst, Mat m);
void Mat_dot(Mat dst, Mat m1, Mat m2);

void Mat_sig(Mat m);

void Mat_free(Mat m);

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

void NN_zero(NN nn);
void NN_rand(NN nn, float low, float high);

void NN_print(NN nn, const char *name);
#define NN_PRINT(nn) NN_print(nn, #nn)

void NN_forward(NN nn);

// ti - train input | to - train output
float NN_cost(NN nn, Mat ti, Mat to);

// gnn - gradient nn | ti - train input | to - train output
void NN_finite_diff(NN nn, NN gnn, Mat ti, Mat to, float eps);

// gnn - gradient nn | ti - train input | to - train output
void NN_backprop(NN nn, NN gnn, Mat ti, Mat to);

// gnn - gradient nn
void NN_learn(NN nn, NN gnn, float rate);

// gnn - gradient nn | ti - train input | to - train output
float NN_accuracy(NN nn, Mat ti, Mat to);

void NN_save(FILE *out, NN nn);
NN NN_load(FILE *in);

void NN_free(NN nn);


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
    Mat m = { 0 };

    m.rows = rows;
    m.cols = cols;

    m.stride = cols;

    m.es = (float*) NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL);

    return m;
}

void Mat_save(FILE* out, Mat m)
{
    const char* magic = "nn.h.mat";
    fwrite(magic, 1, strlen(magic), out);

    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);

    for (size_t i = 0; i < m.rows; i++)
    {
        const float* row = &MAT_AT(m, i, 0);

        size_t written = 0;
        while (written < m.cols)
            written += fwrite(row + written, sizeof(*row), m.cols - written, out);
    }
}

Mat Mat_load(FILE* in)
{
    char magic[8];
    fread(magic, 1, 8, in);
    NN_ASSERT(memcmp(magic, "nn.h.mat", 8) == 0);

    size_t rows, cols;
    fread(&rows, sizeof(rows), 1, in);
    fread(&cols, sizeof(cols), 1, in);

    Mat m = Mat_alloc(rows, cols);

    size_t n = fread(m.es, sizeof(*m.es), rows * cols, in);

    while (n < rows * cols && !ferror(in))
        n += fread(m.es + n, sizeof(*m.es), rows * cols - n, in);

    return m;
}

void Mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t ii = 0; ii < m.cols; ii++)
            MAT_AT(m, i, ii) = rand_float() * (high - low) + low;
}

void Mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t ii = 0; ii < m.cols; ii++)
            MAT_AT(m, i, ii) = x;
}

void Mat_print(Mat m, const char *name, size_t padding)
{
    printf("\n%*s%s = [\n", (int) padding, "", name);

    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t ii = 0; ii < m.cols; ii++)
            printf("%*s    %f", (int) padding, "", MAT_AT(m, i, ii));

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

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t ii = 0; ii < dst.cols; ii++)
            MAT_AT(dst, i, ii) = MAT_AT(src, i, ii);
}

void Mat_sum(Mat dst, Mat m)
{
    NN_ASSERT(dst.rows == m.rows);
    NN_ASSERT(dst.cols == m.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t ii = 0; ii < dst.cols; ii++)
            MAT_AT(dst, i, ii) += MAT_AT(m, i, ii);
}

void Mat_dot(Mat dst, Mat m1, Mat m2)
{
    NN_ASSERT(m1.cols == m2.rows);
    NN_ASSERT(dst.rows == m1.rows);
    NN_ASSERT(dst.cols == m2.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t ii = 0; ii < dst.cols; ii++)
        {
            MAT_AT(dst, i, ii) = 0;

            for (size_t iii = 0; iii < m1.cols; iii++)
                MAT_AT(dst, i, ii) += MAT_AT(m1, i, iii) * MAT_AT(m2, iii, ii);
        }
}

void Mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t ii = 0; ii < m.cols; ii++)
            MAT_AT(m, i, ii) = sigmoidf(MAT_AT(m, i, ii));
}

void Mat_free(Mat m)
{
    if (m.es != NULL)
        free(m.es);
}

NN NN_alloc(size_t* arch, size_t arch_count)
{
    NN_ASSERT(arch_count >= 2);

    NN nn = { 0 };

    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL);

    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL);

    nn.as = NN_MALLOC(sizeof(*nn.as) * arch_count);
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = Mat_alloc(1, arch[0]);

    for (size_t i = 0; i < nn.count; i++)
    {
        NN_ASSERT(i + 1 < arch_count);

        nn.ws[i]     = Mat_alloc(arch[i], arch[i + 1]);
        nn.bs[i]     = Mat_alloc(1, arch[i + 1]);
        nn.as[i + 1] = Mat_alloc(1, arch[i + 1]);
    }

    return nn;
}

void NN_zero(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_fill(nn.ws[i], 0);
        Mat_fill(nn.bs[i], 0);
        Mat_fill(nn.as[i], 0);
    }

    Mat_fill(nn.as[nn.count], 0);
}

void NN_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_rand(nn.ws[i], low, high);
        Mat_rand(nn.bs[i], low, high);
    }
}

void NN_print(NN nn, const char *name)
{
    char buf[256];

    printf("\n%s = [", name);
    
    for (size_t i = 0; i < nn.count; i++)
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
    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        Mat_sum(nn.as[i + 1], nn.bs[i]);
        Mat_sig(nn.as[i + 1]);
    }
}

float NN_cost(NN nn, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    
    float c = 0.0F;

    for (size_t i = 0; i < ti.rows; i++)
    {
        Mat x = Mat_row(ti, i);
        Mat y = Mat_row(to, i);
        
        Mat_copy(NN_INPUT(nn), x);
        NN_forward(nn);

        for (size_t ii = 0; ii < to.cols; ii++)
        {
            float d = MAT_AT(NN_OUTPUT(nn), 0, ii) - MAT_AT(y, 0, ii);

            c += d * d;
        }
    }

    return c / ti.rows;
}

void NN_finite_diff(NN nn, NN gnn, Mat ti, Mat to, float eps)
{
    float saved;

    float c = NN_cost(nn, ti, to);

    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t ii = 0; ii < nn.ws[i].rows; ii++)
            for (size_t iii = 0; iii < nn.ws[i].cols; iii++)
            {
                saved = MAT_AT(nn.ws[i], ii, iii);

                MAT_AT(nn.ws[i], ii, iii) += eps;
                MAT_AT(gnn.ws[i], ii, iii) = (NN_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.ws[i], ii, iii) = saved;
            }

        for (size_t ii = 0; ii < nn.bs[i].rows; ii++)
            for (size_t iii = 0; iii < nn.bs[i].cols; iii++)
            {
                saved = MAT_AT(nn.bs[i], ii, iii);
    
                MAT_AT(nn.bs[i], ii, iii) += eps;
                MAT_AT(gnn.bs[i], ii, iii) = (NN_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.bs[i], ii, iii) = saved;
            }
    }
}

void NN_backprop(NN nn, NN gnn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    NN_zero(gnn);

    /* 
     * i   - current sample
     * iv  - current layer
     * ii  - current activation
     * iii - previous activation
     */

    for (size_t i = 0; i < ti.rows; i++)
    {
        Mat_copy(NN_INPUT(nn), Mat_row(ti, i));

        NN_forward(nn);

        for (size_t ii = 0; ii <= nn.count; ii++)
            Mat_fill(gnn.as[ii], 0);

        for (size_t ii = 0; ii < to.cols; ii++)
            MAT_AT(NN_OUTPUT(gnn), 0, ii) = MAT_AT(NN_OUTPUT(nn), 0, ii) - MAT_AT(to, i, ii);

        for (size_t iv = nn.count; iv > 0; iv--)
            for (size_t ii = 0; ii < nn.as[iv].cols; ii++)
            {
                float a  = MAT_AT(nn.as[iv], 0, ii);
                float da = MAT_AT(gnn.as[iv], 0, ii);

                float dz = 2 * da * a * (1 - a);

                MAT_AT(gnn.bs[iv - 1], 0, ii) += dz;

                for (size_t iii = 0; iii < nn.as[iv - 1].cols; iii++)
                {
                    /*
                     * ii  - weight matrix col
                     * iii - weight matrix row
                     */

                    float pa = MAT_AT(nn.as[iv - 1], 0, iii);
                    float w  = MAT_AT(nn.ws[iv - 1], iii, ii);

                    MAT_AT(gnn.ws[iv - 1], iii, ii) += dz * pa;
                    MAT_AT(gnn.as[iv - 1], 0, iii)  += dz * w;
                }
            }
    }

    for (size_t i = 0; i < gnn.count; i++)
    {
        for (size_t ii = 0; ii < gnn.ws[i].rows; ii++)
            for (size_t iii = 0; iii < gnn.ws[i].cols; iii++)
                MAT_AT(gnn.ws[i], ii, iii) /= ti.rows;

        for (size_t ii = 0; ii < gnn.bs[i].rows; ii++)
            for (size_t iii = 0; iii < gnn.bs[i].cols; iii++)
                MAT_AT(gnn.bs[i], ii, iii) /= ti.rows;
    }
}

void NN_learn(NN nn, NN gnn, float rate)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t ii = 0; ii < nn.ws[i].rows; ii++)
            for (size_t iii = 0; iii < nn.ws[i].cols; iii++)
                MAT_AT(nn.ws[i], ii, iii) -= MAT_AT(gnn.ws[i], ii, iii) * rate;

        for (size_t ii = 0; ii < nn.bs[i].rows; ii++)
            for (size_t iii = 0; iii < nn.bs[i].cols; iii++)
                MAT_AT(nn.bs[i], ii, iii) -= MAT_AT(gnn.bs[i], ii, iii) * rate;
    }
}

float NN_accuracy(NN nn, Mat ti, Mat to)
{
    size_t correct = 0;

    for (size_t i = 0; i < ti.rows; i++)
    {
        Mat x = Mat_row(ti, i);
        Mat y = Mat_row(to, i);

        Mat_copy(NN_INPUT(nn), x);
        NN_forward(nn);

        size_t pred = 0;
        float max_pred = MAT_AT(NN_OUTPUT(nn), 0, 0);

        for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
        {
            float v = MAT_AT(NN_OUTPUT(nn), 0, j);

            if (v > max_pred)
            {
                max_pred = v;
                pred = j;
            }
        }

        size_t label = 0;

        for (size_t j = 0; j < y.cols; j++)
        {
            if (MAT_AT(y, 0, j) == 1.0f)
            {
                label = j;

                break;
            }
        }

        if (pred == label)
            correct++;
    }

    return (float)correct / (float)ti.rows;
}

void NN_save(FILE *out, NN nn)
{
    fwrite(&nn.count, sizeof(nn.count), 1, out);

    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_save(out, nn.ws[i]);
        Mat_save(out, nn.bs[i]);
    }
}

NN NN_load(FILE *in)
{
    size_t count;
    fread(&count, sizeof(count), 1, in);

    NN_ASSERT(count >= 1);

    size_t input_size;
    fread(&input_size, sizeof(input_size), 1, in);


    size_t *arch = (size_t*) malloc(sizeof(size_t) * (count + 1));
    NN_ASSERT(arch != NULL);

    arch[0] = input_size;

    Mat* ws = (Mat*) malloc(sizeof(Mat) * count);
    NN_ASSERT(ws != NULL);

    Mat* bs = (Mat*) malloc(sizeof(Mat) * count);
    NN_ASSERT(bs != NULL);

    for (size_t i = 0; i < count; i++)
    {
        ws[i] = Mat_load(in);
        bs[i] = Mat_load(in);
        arch[i + 1] = ws[i].cols; // assuming ws[i] shape is [prev_layer_size, this_layer_size]
    }

    NN nn = NN_alloc(arch, count + 1);

    for (size_t i = 0; i < count; i++)
    {
        Mat_copy(nn.ws[i], ws[i]);
        Mat_copy(nn.bs[i], bs[i]);

        Mat_free(ws[i]);
        Mat_free(bs[i]);
    }

    free(ws);
    free(bs);
    free(arch);

    return nn;
}

void NN_free(NN nn)
{
    if (nn.count > 0)
        for (size_t i = 0; i < nn.count; i++)
        {
            Mat_free(nn.ws[i]);
            Mat_free(nn.bs[i]);
            Mat_free(nn.as[i + 1]);
        }

    Mat_free(nn.as[0]);

    if (nn.ws != NULL) free(nn.ws);
    if (nn.bs != NULL) free(nn.bs);
    if (nn.as != NULL) free(nn.as);
}