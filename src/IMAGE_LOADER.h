#pragma once

#include <stdio.h>
#include <stdlib.h>

// Load MNIST images from file path
// Returns float* pointer to images in [count, 784] format normalized [0,1]
// count_out: number of images loaded
float* load_mnist_images(const char *path, int *count_out);

// Load MNIST labels from file path
// Returns uint8_t* pointer to labels
// count_out: number of labels loaded
uint8_t* load_mnist_labels(const char *path, int *count_out);

static uint32_t reverse_int(uint32_t i)
{
    return ((i & 0xFF) << 24) |
           ((i & 0xFF00) << 8) |
           ((i & 0xFF0000) >> 8) |
           ((i & 0xFF000000) >> 24);
}

float* load_mnist_images(const char *path, int *count_out)
{
    FILE *file = fopen(path, "rb");

    if (!file)
    {
        fprintf(stderr, "Error: cannot open image file %s\n", path);

        return NULL;
    }

    uint32_t magic = 0;
    fread(&magic, sizeof(magic), 1, file);
    magic = reverse_int(magic);

    if (magic != 2051)
    {
        fprintf(stderr, "Error: invalid magic number in image file\n");

        fclose(file);

        return NULL;
    }

    uint32_t num_images = 0;
    fread(&num_images, sizeof(num_images), 1, file);
    num_images = reverse_int(num_images);

    uint32_t rows = 0;
    fread(&rows, sizeof(rows), 1, file);
    rows = reverse_int(rows);

    uint32_t cols = 0;
    fread(&cols, sizeof(cols), 1, file);
    cols = reverse_int(cols);

    if (rows != 28 || cols != 28)
    {
        fprintf(stderr, "Error: expected 28x28 images but got %ux%u\n", rows, cols);

        fclose(file);

        return NULL;
    }

    *count_out = (int)num_images;

    size_t img_size = rows * cols;

    float *images = (float*)malloc(sizeof(float) * img_size * num_images);

    if (!images)
    {
        fprintf(stderr, "Error: failed to allocate memory for images\n");

        fclose(file);

        return NULL;
    }

    for (uint32_t i = 0; i < num_images; i++)
    {
        unsigned char buffer[28 * 28];

        if (fread(buffer, sizeof(unsigned char), img_size, file) != img_size)
        {
            fprintf(stderr, "Error: failed to read image %u\n", i);

            free(images);
            fclose(file);

            return NULL;
        }

        // Normalize pixel values to [0, 1]
        for (size_t j = 0; j < img_size; j++)
            images[i * img_size + j] = buffer[j] / 255.0f;
    }

    fclose(file);

    return images;
}

uint8_t* load_mnist_labels(const char *path, int *count_out)
{
    FILE *file = fopen(path, "rb");

    if (!file)
    {
        fprintf(stderr, "Error: cannot open label file %s\n", path);

        return NULL;
    }

    uint32_t magic = 0;
    fread(&magic, sizeof(magic), 1, file);
    magic = reverse_int(magic);

    if (magic != 2049)
    {
        fprintf(stderr, "Error: invalid magic number in label file\n");

        fclose(file);

        return NULL;
    }

    uint32_t num_labels = 0;
    fread(&num_labels, sizeof(num_labels), 1, file);
    num_labels = reverse_int(num_labels);

    *count_out = (int)num_labels;

    uint8_t *labels = (uint8_t*)malloc(sizeof(uint8_t) * num_labels);

    if (!labels)
    {
        fprintf(stderr, "Error: failed to allocate memory for labels\n");

        fclose(file);

        return NULL;
    }

    if (fread(labels, sizeof(uint8_t), num_labels, file) != num_labels)
    {
        fprintf(stderr, "Error: failed to read labels\n");

        free(labels);
        fclose(file);

        return NULL;
    }

    fclose(file);

    return labels;
}