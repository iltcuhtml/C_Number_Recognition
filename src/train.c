#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "NN.h"

#define CELL_LEN 28

int main()
{
    // --- Load dataset ---
    FILE* file = NULL;

    if (fopen_s(&file, "data/number.dat", "rb") != 0 || !file)
    {
        printf("Failed to open 'data/number.dat'\n");

        return EXIT_FAILURE;
    }

    char header[7];

    if (fread(header, sizeof(char), 7, file) != 7)
    {
        printf("Failed to read file header\n");

        fclose(file);

        return EXIT_FAILURE;
    }

    if (memcmp(header, "NUMDATA", 7) != 0)
    {
        fclose(file);

        printf("File header mismatch");

        return EXIT_FAILURE;
    }

    size_t sample_count = 0;

    if (fread(&sample_count, sizeof(size_t), 1, file) != 1 || sample_count == 0)
    {
        fclose(file);

        printf("No data in 'data/number.dat'\n");

        return EXIT_FAILURE;
    }

    printf("Dataset loaded, %zu samples\n", sample_count);

    const int input_size = CELL_LEN * CELL_LEN;
    const int num_classes = 10;

    // Load images
    uint8_t* raw_images = (uint8_t*)malloc(sizeof(uint8_t) * sample_count * input_size);

    if (!raw_images)
    {
        printf("Memory allocation failed\n");

        fclose(file);

        return EXIT_FAILURE;
    }

    if (fread(raw_images, sizeof(uint8_t), sample_count * input_size, file) != sample_count * input_size)
    {
        printf("Failed to read image data\n");

        free(raw_images);

        fclose(file);

        return EXIT_FAILURE;
    }

    fclose(file);

    Mat train_inputs = Mat_alloc(sample_count, input_size);

    for (size_t i = 0; i < sample_count; i++)
        for (size_t j = 0; j < input_size; j++)
            MAT_AT(train_inputs, i, j) = raw_images[i * input_size + j] / 255.0f;

    free(raw_images);

    // Generate one-hot labels
    Mat train_labels = Mat_alloc(sample_count, num_classes);

    for (size_t i = 0; i < sample_count; i++)
    {
        Mat row = Mat_row(train_labels, i);

        for (int j = 0; j < num_classes; j++)
            MAT_AT(row, 0, j) = 0.0f;

        MAT_AT(row, 0, i % num_classes) = 1.0f;
    }
    
    // --- Network & ConvLayer ---
    ConvLayer conv = Conv_alloc(1, 16, 3);

    size_t conv_out_rows = CELL_LEN - conv.kernel_size + 1;
    size_t pool_rows = conv_out_rows / 2;

    size_t fc_input_size = conv.out_channels * pool_rows * pool_rows;
    size_t fc_arch[] = { fc_input_size, 128, 64, num_classes };
	
    const float lr = 0.05f;
    const int epochs = 5000;

    // Allocate NN and gradient
    NN nn = NN_alloc(fc_arch, sizeof(fc_arch) / sizeof(*fc_arch));
    NN grad_fc = NN_alloc(fc_arch, sizeof(fc_arch) / sizeof(*fc_arch));
	
    // Xavier init for FC
    NN_xavier_init(nn);

    // Bias small positive
    for (size_t i = 0; i < nn.count; i++)
        Mat_fill(nn.bs[i], 0.01f);

    // Conv layer init (only kernels & biases)
    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        Mat_rand(conv.kernels[i], -0.1f, 0.1f);

    for (size_t i = 0; i < conv.out_channels; i++)
        MAT_AT(conv.biases[i], 0, 0) = 0.0f;

    // --- Training loop ---
    for (int e = 1; e <= epochs; e++)
        CNN_train_epoch(nn, grad_fc, &conv, train_inputs, train_labels, lr, e, epochs);
	
    // --- Save model ---
    FILE* out_file = NULL;
    fopen_s(&out_file, "data/model.cnn", "wb");

    if (out_file)
    {
        CNN_save(out_file, conv, nn);   // Save both ConvLayer and FC NN
        
        fclose(out_file);
        
        printf("Model saved to 'data/model.cnn'\n");
    }
    else
    {
        printf("Failed to open 'data/model.cnn' for saving\n");
    }

    // --- Cleanup ---
    NN_free(&nn);
    NN_free(&grad_fc);

    Mat_free(train_inputs);
    Mat_free(train_labels);

    Conv_free(&conv);
}