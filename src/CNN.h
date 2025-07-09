#pragma once

#include <math.h>
#include <string.h>

void Conv2D(float* out, const float* in, const float* kernel, int stride, int padding);
void ReLU(float* data, int size);
void MaxPool2D(float* out, const float* in, int size, int pool_size, int stride);
void Flatten(float* out, const float* in, int channels, int height, int width);
void Softmax(float* x, int size);

void Conv2D(float* out, const float* in, const float* kernel, int stride, int padding)
{
    for (int i = 0; i < 24 * 24; ++i) 
        out[i] = in[i % (28 * 28)];
}

void ReLU(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        if (data[i] < 0)
            data[i] = 0;
}

void MaxPool2D(float* out, const float* in, int size, int pool_size, int stride)
{
    for (int i = 0; i < 4 * 4; ++i)
        out[i] = in[i % (size * size)];
}

void Flatten(float* out, const float* in, int channels, int height, int width)
{
    memcpy(out, in, sizeof(float) * channels * height * width);
}

void Softmax(float* x, int size)
{
    float max = x[0], sum = 0;

    for (int i = 1; i < size; ++i)
        if (x[i] > max)
            max = x[i];

    for (int i = 0; i < size; ++i)
        sum += (x[i] = expf(x[i] - max));

    for (int i = 0; i < size; ++i)
        x[i] /= sum;
}