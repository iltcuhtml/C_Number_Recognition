#include <stdio.h>

#include "CNN.h"
#include "NN.h"
#include "IMAGE_LOADER.h"

int main()
{
    float input_pixels[28*28];
    load_pgm("data/user_input.pgm", input_pixels, 28*28);

    Mat input = (Mat) {
        .rows = 28, .cols = 28, .stride = 28, .es = input_pixels
    };

    Mat kernel = Mat_load(fopen("model.cnn", "rb"));
    Mat weights = Mat_load(fopen("model.cnn", "rb"));
    Mat bias = Mat_load(fopen("model.cnn", "rb"));

    Mat conv_out = Mat_alloc(26, 26);
    Mat pooled = Mat_alloc(13, 13);
    Mat flat = Mat_alloc(1, 13*13);
    Mat out = Mat_alloc(1, 10);

    Conv2D(conv_out, input, kernel, 1, 0);
    MaxPool2D(pooled, conv_out, 2, 2, 2);
    Flatten(flat, &pooled, 1, 13, 13);
    Mat_dot(out, flat, weights);
    Mat_sum(out, bias);
    Softmax(out);

    printf("Prediction: \n");

    for (int i = 0; i < 10; i++)
        printf("[%d] = %.2f\n", i, MAT_AT(out, 0, i));

    Mat_free(kernel);
    Mat_free(weights);
    Mat_free(bias);
    Mat_free(conv_out);
    Mat_free(pooled);
    Mat_free(flat);
    Mat_free(out);

    return EXIT_SUCCESS;
}