#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string.h>

// -------------------------
// Utilities
// -------------------------
float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float dsigmoid(float y)
{
    return y * (1.0f - y);
}

float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

float drelu(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

// -------------------------
// Matrix
// -------------------------
typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;

    float* es;
} Mat;

#define MAT_AT(m,r,c) (m).es[(r)*(m).stride + (c)]

Mat Mat_alloc(size_t rows, size_t cols)
{
    Mat m = { rows, cols, cols, malloc(sizeof(float) * rows * cols) };

    assert(m.es != NULL);

    return m;
}

void Mat_free(Mat m)
{
    if (m.es)
        free(m.es);
}

void Mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = x;
}

void Mat_copy(Mat dst, Mat src)
{
    assert(dst.rows == src.rows && dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
}

void Mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
}

void Mat_dot(Mat dst, Mat m1, Mat m2)
{
    assert(m1.cols == m2.rows && dst.rows == m1.rows && dst.cols == m2.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            for (size_t k = 0; k < m1.cols; k++)
                MAT_AT(dst, i, j) += MAT_AT(m1, i, k) * MAT_AT(m2, k, j);
}

void Mat_sum(Mat dst, Mat m)
{
    assert(dst.rows == m.rows && dst.cols == m.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            MAT_AT(dst, i, j) += MAT_AT(m, i, j);
}

void Mat_relu_inplace(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            if (MAT_AT(m, i, j) < 0)
                MAT_AT(m, i, j) = 0;
}

void Mat_save(FILE* out, Mat m)
{
    fwrite("MATDATA", sizeof(char), 7, out);

    fwrite(&m.rows, sizeof(size_t), 1, out);
    fwrite(&m.cols, sizeof(size_t), 1, out);

    fwrite(m.es, sizeof(float), m.rows * m.cols, out);
}

Mat Mat_load(FILE* in)
{
    char header[7];
    fread(header, sizeof(char), 7, in);
    assert(memcmp(header, "MATDATA", 7) == 0);

    size_t rows, cols;
    fread(&rows, sizeof(size_t), 1, in);
    fread(&cols, sizeof(size_t), 1, in);

    Mat m = Mat_alloc(rows, cols);
    fread(m.es, sizeof(float), rows * cols, in);

    return m;
}

// -------------------------
// Convolution Layer
// -------------------------
typedef struct
{
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;

    Mat* kernels;
    Mat* biases;
} ConvLayer;

ConvLayer Conv_alloc(size_t in_channels, size_t out_channels, size_t kernel_size)
{
    ConvLayer conv = { 0 };

    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel_size = kernel_size;

    conv.kernels = malloc(sizeof(Mat) * in_channels * out_channels);
    conv.biases = malloc(sizeof(Mat) * out_channels);

    for (size_t oc = 0; oc < out_channels; oc++)
    {
        conv.biases[oc] = Mat_alloc(1, 1);
        MAT_AT(conv.biases[oc], 0, 0) = 0;

        for (size_t ic = 0; ic < in_channels; ic++)
            conv.kernels[oc * in_channels + ic] = Mat_alloc(kernel_size, kernel_size);
    }

    return conv;
}

void Conv_forward_single(Mat kernel, Mat input, Mat* output)
{
    size_t out_rows = input.rows - kernel.rows + 1;
    size_t out_cols = input.cols - kernel.cols + 1;

    *output = Mat_alloc(out_rows, out_cols);

    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
        {
            float sum = 0.0f;

            for (size_t ky = 0; ky < kernel.rows; ky++)
                for (size_t kx = 0; kx < kernel.cols; kx++)
                    sum += MAT_AT(input, y + ky, x + kx) * MAT_AT(kernel, ky, kx);

            MAT_AT(*output, y, x) = sum;
        }

    Mat_relu_inplace(*output);
}

void Conv_forward(ConvLayer conv, Mat* input_channels, Mat* output_channels)
{
    for (size_t oc = 0; oc < conv.out_channels; oc++)
    {
        output_channels[oc] = Mat_alloc(
            input_channels[0].rows - conv.kernel_size + 1,
            input_channels[0].cols - conv.kernel_size + 1
        );

        Mat_fill(output_channels[oc], MAT_AT(conv.biases[oc], 0, 0));

        for (size_t ic = 0; ic < conv.in_channels; ic++)
        {
            Mat temp;

            Conv_forward_single(conv.kernels[oc * conv.in_channels + ic], input_channels[ic], &temp);
            Mat_sum(output_channels[oc], temp);
            Mat_free(temp);
        }

        Mat_relu_inplace(output_channels[oc]);
    }
}

void Conv_save(FILE* out, ConvLayer conv)
{
    fwrite("CONVLAYR", sizeof(char), 8, out);

    fwrite(&conv.in_channels, sizeof(size_t), 1, out);
    fwrite(&conv.out_channels, sizeof(size_t), 1, out);
    fwrite(&conv.kernel_size, sizeof(size_t), 1, out);

    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        Mat_save(out, conv.kernels[i]);

    for (size_t i = 0; i < conv.out_channels; i++)
        Mat_save(out, conv.biases[i]);
}

ConvLayer Conv_load(FILE* in)
{
    char header[8];
    fread(header, sizeof(char), 8, in);
    assert(memcmp(header, "CONVLAYR", 8) == 0);

    ConvLayer conv = { 0 };

    fread(&conv.in_channels, sizeof(size_t), 1, in);
    fread(&conv.out_channels, sizeof(size_t), 1, in);
    fread(&conv.kernel_size, sizeof(size_t), 1, in);

    conv.kernels = malloc(sizeof(Mat) * conv.in_channels * conv.out_channels);
    conv.biases = malloc(sizeof(Mat) * conv.out_channels);

    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        conv.kernels[i] = Mat_load(in);

    for (size_t i = 0; i < conv.out_channels; i++)
        conv.biases[i] = Mat_load(in);

    return conv;
}

// -------------------------
// MaxPool Layer
// -------------------------
Mat MaxPool2D(Mat input, size_t pool_size, size_t stride)
{
    size_t out_rows = (input.rows - pool_size) / stride + 1;
    size_t out_cols = (input.cols - pool_size) / stride + 1;

    Mat out = Mat_alloc(out_rows, out_cols);

    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
        {
            float max_val = -FLT_MAX;

            for (size_t py = 0; py < pool_size; py++)
                for (size_t px = 0; px < pool_size; px++)
                {
                    float v = MAT_AT(input, y * stride + py, x * stride + px);

                    if (v > max_val)
                        max_val = v;
                }

            MAT_AT(out, y, x) = max_val;
        }

    return out;
}

// -------------------------
// Flatten
// -------------------------
Mat Flatten(Mat* channels, size_t channel_count)
{
    size_t total = 0;

    for (size_t c = 0; c < channel_count; c++)
        total += channels[c].rows * channels[c].cols;

    Mat out = Mat_alloc(1, total);

    size_t idx = 0;

    for (size_t c = 0; c < channel_count; c++)
        for (size_t i = 0; i < channels[c].rows; i++)
            for (size_t j = 0; j < channels[c].cols; j++)
                MAT_AT(out, 0, idx++) = MAT_AT(channels[c], i, j);

    return out;
}

// -------------------------
// Fully Connected NN
// -------------------------
typedef struct
{
    size_t count;

    Mat* ws;
    Mat* bs;
    Mat* as;

    size_t conv_count;
    ConvLayer* convs;
} NN;


#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN NN_alloc(size_t* arch, size_t arch_count)
{
    assert(arch_count >= 2);

    NN nn = { 0 };

    nn.count = arch_count - 1;

    nn.ws = malloc(sizeof(Mat) * nn.count);
    nn.bs = malloc(sizeof(Mat) * nn.count);
    nn.as = malloc(sizeof(Mat) * arch_count);

    nn.as[0] = Mat_alloc(1, arch[0]);

    for (size_t i = 0; i < nn.count; i++)
    {
        nn.ws[i] = Mat_alloc(arch[i], arch[i + 1]);
        nn.bs[i] = Mat_alloc(1, arch[i + 1]);
        nn.as[i + 1] = Mat_alloc(1, arch[i + 1]);
    }

    return nn;
}

void NN_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_rand(nn.ws[i], low, high);
        Mat_rand(nn.bs[i], low, high);
    }
}

void NN_forward(NN nn)
{
    Mat x = NN_INPUT(nn);

    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_fill(nn.as[i + 1], 0);
        Mat_dot(nn.as[i + 1], x, nn.ws[i]);
        Mat_sum(nn.as[i + 1], nn.bs[i]);
        Mat_relu_inplace(nn.as[i + 1]);

        x = nn.as[i + 1];
    }
}

void NN_save(FILE* out, NN nn)
{
    fwrite("NNMODEL", sizeof(char), 7, out);

    fwrite(&nn.count, sizeof(size_t), 1, out);
    fwrite(&nn.conv_count, sizeof(size_t), 1, out);

    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_save(out, nn.ws[i]);
        Mat_save(out, nn.bs[i]);
    }

    for (size_t i = 0; i < nn.conv_count; i++)
        Conv_save(out, nn.convs[i]);
}

NN NN_load(FILE* in)
{
    char header[7];
    fread(header, sizeof(char), 7, in);
    assert(memcmp(header, "NNMODEL", 7) == 0);

    NN nn = { 0 };

    fread(&nn.count, sizeof(size_t), 1, in);
    fread(&nn.conv_count, sizeof(size_t), 1, in);

    size_t* arch = malloc(sizeof(size_t) * (nn.count + 1));

    nn.ws = malloc(sizeof(Mat) * nn.count);
    nn.bs = malloc(sizeof(Mat) * nn.count);
    nn.as = malloc(sizeof(Mat) * (nn.count + 1));

    for (size_t i = 0; i < nn.count; i++)
    {
        nn.ws[i] = Mat_load(in);
        nn.bs[i] = Mat_load(in);
        arch[i + 1] = nn.ws[i].cols;
    }

    arch[0] = nn.ws[0].rows;
    nn.as[0] = Mat_alloc(1, arch[0]);

    for (size_t i = 0; i < nn.count; i++)
        nn.as[i + 1] = Mat_alloc(1, arch[i + 1]);

    if (nn.conv_count > 0)
    {
        nn.convs = malloc(sizeof(ConvLayer) * nn.conv_count);

        for (size_t i = 0; i < nn.conv_count; i++)
            nn.convs[i] = Conv_load(in);
    }

    free(arch);

    return nn;
}