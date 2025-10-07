#include <stdio.h>
#include <stdlib.h>
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

    char header[8];
    fread(header, sizeof(char), 8, file);

    if (memcmp(header, "nn.h.dat", 8) != 0)
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

    uint8_t* raw_images = (uint8_t*)malloc(sizeof(uint8_t) * sample_count * input_size);

    if (!raw_images)
    {
        fclose(file);

        printf("Memory allocation failed\n");

        return EXIT_FAILURE;
    }

    size_t read_bytes = fread(raw_images, sizeof(uint8_t), sample_count * input_size, file);

    fclose(file);

    if (read_bytes != sample_count * input_size)
    {
        free(raw_images);

        printf("Failed to read all image data\n");

        return EXIT_FAILURE;
    }

    Mat train_inputs = Mat_alloc(sample_count, input_size);

    for (int i = 0; i < sample_count; i++)
        for (int j = 0; j < input_size; j++)
            MAT_AT(train_inputs, i, j) = raw_images[i * input_size + j] / 255.0f;

	free(raw_images);

    Mat train_labels = Mat_alloc(sample_count, num_classes);

    for (size_t i = 0; i < sample_count; i++)
    {
        Mat row = Mat_row(train_labels, i);

        for (int j = 0; j < num_classes; j++)
            MAT_AT(row, 0, j) = 0.0f;

        MAT_AT(row, 0, i % 10) = 1.0f;
    }

    size_t arch[] = { input_size, 16, 16, num_classes };
    NN nn = NN_alloc(arch, sizeof(arch) / sizeof(*arch));
    NN gnn = NN_alloc(arch, sizeof(arch) / sizeof(*arch));

    NN_rand(nn, -1.0f, 1.0f);

    const int epochs = 50000;
    const float learning_rate = 0.01f;

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        NN_backprop(nn, gnn, train_inputs, train_labels);
        NN_learn(nn, gnn, learning_rate);

        float cost = NN_cost(nn, train_inputs, train_labels);
        float acc = NN_accuracy(nn, train_inputs, train_labels);

        printf("Epoch %d, cost = %.4f, accuracy = %.4f\n", epoch, cost, acc);
    }

    FILE* out_file = NULL;

    fopen_s(&out_file, "data/model.nn", "wb");

    if (out_file)
    {
        NN_save(out_file, nn);

        fclose(out_file);

        printf("Model saved to 'data/model.nn'\n");
    }

    NN_free(nn);
    NN_free(gnn);
    Mat_free(train_inputs);
    Mat_free(train_labels);

    return EXIT_SUCCESS;
}

int main2()
{
    const int input_size = 28 * 28;
    const int num_classes = 10;
    const size_t sample_count = 1000;

    Mat train_inputs = Mat_alloc(sample_count, input_size);
    Mat train_labels = Mat_alloc(sample_count, num_classes);

    for (size_t i = 0; i < sample_count; i++)
        for (size_t j = 0; j < input_size; j++)
            MAT_AT(train_inputs, i, j) = rand_float();

    for (size_t i = 0; i < sample_count; i++)
    {
        for (size_t j = 0; j < num_classes; j++)
            MAT_AT(train_labels, i, j) = 0;

        MAT_AT(train_labels, i, i % num_classes) = 1;
    }

    size_t arch[] = { input_size, 128, 64, num_classes };
    NN nn = NN_alloc(arch, sizeof(arch) / sizeof(*arch));
    NN_rand(nn, -1.0f, 1.0f);

    const int epochs = 10;
    const float lr = 0.01f;

    for (int e = 0; e < epochs; e++)
        NN_forward(nn);

    Mat_free(train_inputs);
    Mat_free(train_labels);

    return 0;
}