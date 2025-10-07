#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "NN.h"
#include "image_loader.h"

int main()
{
    const int input_size = 28 * 28;
    const int num_classes = 10;
    const int sample_count = image_count;

    // Prepare input Mat
    Mat train_inputs = Mat_alloc(sample_count, input_size);

    // Copy raw image data into train_inputs
    for (int i = 0; i < sample_count; i++)
        for (int j = 0; j < input_size; j++)
            MAT_AT(train_inputs, i, j) = raw_images[i * input_size + j];

    // Prepare output Mat (one-hot labels)
    Mat train_labels = Mat_alloc(sample_count, num_classes);
    labels_to_onehot(train_labels, raw_labels, sample_count, num_classes);

    free(raw_images);
    free(raw_labels);

    // Neural network architecture: input -> hidden -> output
    size_t arch[] = { input_size, 32, num_classes };
    NN nn = NN_alloc(arch, sizeof(arch) / sizeof(*arch));
    NN gnn = NN_alloc(arch, sizeof(arch) / sizeof(*arch));

    NN_rand(nn, -1.0f, 1.0f);

    const int epochs = 10;
    const float learning_rate = 0.5f;

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        NN_backprop(nn, gnn, train_inputs, train_labels);
        NN_learn(nn, gnn, learning_rate);

        if (epoch % 10 == 0)
        {
            float cost = NN_cost(nn, train_inputs, train_labels);
            float acc = NN_accuracy(nn, train_inputs, train_labels);

            printf("Epoch %d, cost = %.4f, accuracy = %.4f\n", epoch, cost, acc);
        }
    }

    FILE *file = fopen("model.nn", "wb");

    if (!file)
    {
        fprintf(stderr, "Failed to open file for saving model\n");
    }
    else
    {
        NN_save(file, nn);
        fclose(file);

        printf("Model saved to model.nn\n");
    }

    // Free memory
    NN_free(nn);
    NN_free(gnn);
    Mat_free(train_inputs);
    Mat_free(train_labels);

    return 0;
}