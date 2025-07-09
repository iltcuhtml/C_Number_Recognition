#ifndef CNN_H
#define CNN_H

#include <stdlib.h>
#include <math.h>
#include <stddef.h>

#include "NN.h"

void Conv2D(Mat out, const Mat in, const Mat kernel,
            size_t stride, size_t padding);

void MaxPool2D(Mat out, const Mat in,
               size_t pool_h, size_t pool_w,
               size_t stride);

void Flatten(Mat out, const Mat *maps, size_t map_count,
             size_t map_h, size_t map_w);

void Softmax(Mat m);

// Convolution: single 2D kernel on single-channel input
void Conv2D(Mat out, const Mat in, const Mat kernel,
            size_t stride, size_t padding)
{
    size_t out_h = out.rows;
    size_t out_w = out.cols;
    size_t k_h = kernel.rows;
    size_t k_w = kernel.cols;

    for (size_t i = 0; i < out_h; i++)
        for (size_t j = 0; j < out_w; j++)
        {
            float sum = 0.0f;

            for (size_t ki = 0; ki < k_h; ki++)
                for (size_t kj = 0; kj < k_w; kj++)
                {
                    int in_i = (int)i * stride + ki - padding;
                    int in_j = (int)j * stride + kj - padding;

                    if (in_i >= 0 && in_i < (int)in.rows &&
                        in_j >= 0 && in_j < (int)in.cols)
                    {
                        sum += MAT_AT(in, in_i, in_j) *
                               MAT_AT(kernel, ki, kj);
                    }
                }

            MAT_AT(out, i, j) = sum;
        }
}

// Max pooling on single-channel input
void MaxPool2D(Mat out, const Mat in,
               size_t pool_h, size_t pool_w,
               size_t stride)
{
    size_t out_h = out.rows;
    size_t out_w = out.cols;

    for (size_t i = 0; i < out_h; i++)
        for (size_t j = 0; j < out_w; j++)
        {
            float max_val = -INFINITY;

            for (size_t ph = 0; ph < pool_h; ph++)
                for (size_t pw = 0; pw < pool_w; pw++)
                {
                    size_t in_i = i * stride + ph;
                    size_t in_j = j * stride + pw;

                    if (in_i < in.rows && in_j < in.cols)
                    {
                        float v = MAT_AT(in, in_i, in_j);

                        if (v > max_val)
                            max_val = v;
                    }
                }
            
            MAT_AT(out, i, j) = max_val;
        }
}

// Flatten N feature maps into one vector
void Flatten(Mat out, const Mat *maps, size_t map_count,
             size_t map_h, size_t map_w)
{
    size_t idx = 0;

    for (size_t m = 0; m < map_count; m++)
        for (size_t i = 0; i < map_h; i++)
            for (size_t j = 0; j < map_w; j++)
                out.es[idx++] = MAT_AT(maps[m], i, j);

    NN_ASSERT(idx == out.cols);
}

// Softmax for output layer
void Softmax(Mat m)
{
    float max = -INFINITY;
    
    for (size_t j = 0; j < m.cols; j++)
        if (MAT_AT(m, 0, j) > max)
            max = MAT_AT(m, 0, j);
    
    float sum = 0.0f;

    for (size_t j = 0; j < m.cols; j++)
    {
        MAT_AT(m, 0, j) = expf(MAT_AT(m, 0, j) - max);

        sum += MAT_AT(m, 0, j);
    }

    for (size_t j = 0; j < m.cols; j++)
        MAT_AT(m, 0, j) /= sum;
}

#endif // CNN_H