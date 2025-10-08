#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "NN.h"

int main()
{
    FILE* file = NULL;
    fopen_s(&file, "data/number.dat", "rb");

    if (!file)
    {
        printf("Failed to open 'data/number.dat'");

        return EXIT_FAILURE;
    }

    char header[7];
    fread(header, sizeof(char), 7, file);

    if (memcmp(header, "NUMDATA", 7) != 0)
    {
        fclose(file);

        printf("File header mismatch");

        return EXIT_FAILURE;
    }

    size_t sample_count = 0;
    fread(&sample_count, sizeof(size_t), 1, file);

    if (sample_count == 0)
    {
        fclose(file);

        printf("No data in 'data/number.dat'");

        return EXIT_FAILURE;
	}

    printf("Dataset loaded, %zu samples\n", sample_count);

    const int input_size = 28 * 28;
    const int num_classes = 10;

    // Load images
    uint8_t* raw_images = (uint8_t*)malloc(sizeof(uint8_t) * sample_count * input_size);
    fread(raw_images, sizeof(uint8_t), sample_count * input_size, file);
    fclose(file);

    Mat train_inputs = Mat_alloc(sample_count, input_size);
    
    for (size_t i = 0; i < sample_count; i++)
        for (size_t j = 0; j < input_size; j++)
            MAT_AT(train_inputs, i, j) = raw_images[i * input_size + j] / 255.0f;

    // Generate one-hot labels
    Mat train_labels = Mat_alloc(sample_count, num_classes);

    for (size_t i = 0; i < sample_count; i++)
    {
        Mat row = Mat_row(train_labels, i);

        for (int j = 0; j < num_classes; j++)
            MAT_AT(row, 0, j) = 0.0f;

        MAT_AT(row, 0, i % num_classes) = 1.0f;
    }

    // CNN + FC NN architecture
    size_t fc_arch[] = { 16 * 13 * 13, 128, 64, num_classes }; // Flattened conv output ¡æ FC layers
    
    NN nn = NN_alloc(fc_arch, sizeof(fc_arch) / sizeof(*fc_arch));
    NN grad = NN_alloc(fc_arch, sizeof(fc_arch) / sizeof(*fc_arch));
    
    NN_rand(nn, -1.0f, 1.0f);

    // Conv layer: 1 input channel, 16 output channels, 3x3 kernel
    ConvLayer conv = Conv_alloc(1, 16, 3);

    const float lr = 0.01f;
    const int epochs = 1000;

    // Temp buffers for conv/pool/flatten
    Mat* conv_out = malloc(sizeof(Mat) * conv.out_channels);
    Mat* pooled = malloc(sizeof(Mat) * conv.out_channels);

    Mat flat;

    for (int e = 1; e <= epochs; e++)
    {
        CNN_train_epoch(nn, grad, conv, train_inputs, train_labels, lr);

        float cost = 0.0f;
        float correct = 0.0f;

        // Compute cost/accuracy on full dataset
        for (size_t i = 0; i < sample_count; i++)
        {
            Mat input_row = Mat_row(train_inputs, i);
            Mat label_row = Mat_row(train_labels, i);

            Mat input_image = Mat_alloc(28, 28);

            for (size_t y = 0; y < 28; y++)
                for (size_t x = 0; x < 28; x++)
                    MAT_AT(input_image, y, x) = MAT_AT(input_row, 0, y * 28 + x);

            CNN_forward_sample(nn, conv, input_image, conv_out, pooled, &flat);

            // Cross-entropy loss
            for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
            {
                float y = MAT_AT(label_row, 0, j);
                float p = MAT_AT(NN_OUTPUT(nn), 0, j);

                cost -= y * logf(fmaxf(p, 1e-7f));
            }

            // Accuracy
            size_t max_idx = 0;

            float max_val = MAT_AT(NN_OUTPUT(nn), 0, 0);
            
            for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
                if (MAT_AT(NN_OUTPUT(nn), 0, j) > max_val)
                {
                    max_val = MAT_AT(NN_OUTPUT(nn), 0, j);
                    max_idx = j;
                }
            
            for (size_t j = 0; j < label_row.cols; j++)
                if (MAT_AT(label_row, 0, j) == 1.0f && j == max_idx)
                    correct++;

            Mat_free(input_image);

            for (size_t c = 0; c < conv.out_channels; c++)
            {
                Mat_free(conv_out[c]);
                Mat_free(pooled[c]);
            }
            
            Mat_free(flat);
        }

        cost /= sample_count;

        float acc = correct / sample_count;

        printf("Epoch %d, cost = %.4f, accuracy = %.4f\n", e, cost, acc);
    }

    FILE* out_file = NULL;

    fopen_s(&out_file, "data/model.nn", "wb");

    if (out_file)
    {
        NN_save(out_file, nn);

        fclose(out_file);

        printf("Model saved to 'data/model.nn'\n");
    }

    NN_free(&nn);
    NN_free(&grad);

    Mat_free(train_inputs);
    Mat_free(train_labels);

    free(conv_out);
    free(pooled);

    return EXIT_SUCCESS;
}