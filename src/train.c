#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "NN.h"
#include "image_loader.h"

// One-hot encode labels into Mat structure
void labels_to_onehot(Mat labels_mat, uint8_t *labels, int count, int num_classes)
{
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < num_classes; j++)
            MAT_AT(labels_mat, i, j) = 0.0f;
        
        int lbl = labels[i];

        if (lbl >= 0 && lbl < num_classes)
            MAT_AT(labels_mat, i, lbl) = 1.0f;
    }
}

int main()
{
    // Paths to MNIST files (adjust paths as needed)
    const char *train_images_path = "data/train-images.idx3-ubyte";
    const char *train_labels_path = "data/train-labels.idx1-ubyte";

    int image_count = 0;
    float *raw_images = load_mnist_images(train_images_path, &image_count);

    if (!raw_images)
    {
        fprintf(stderr, "Failed to load images.\n");

        return EXIT_FAILURE;
    }

    int label_count = 0;
    uint8_t *raw_labels = load_mnist_labels(train_labels_path, &label_count);

    if (!raw_labels)
    {
        fprintf(stderr, "Failed to load labels.\n");

        free(raw_images);

        return EXIT_FAILURE;
    }

    if (image_count != label_count)
    {
        fprintf(stderr, "Image and label counts do not match.\n");

        free(raw_images);
        free(raw_labels);

        return EXIT_FAILURE;
    }

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

    const int epochs = 1000;
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