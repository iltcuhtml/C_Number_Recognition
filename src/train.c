#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "CNN.h"
#include "NN.h"
#include "IMAGE_LOADER.h"

int main()
{
    int count, rows, cols;
    unsigned char* images = load_idx_images("data/train-images.idx3-ubyte", &count, &rows, &cols);
    unsigned char* labels = load_idx_labels("data/train-labels.idx1-ubyte", &count);

    Mat kernel = Mat_alloc(3, 3); // single 3x3 conv kernel
    Mat weights = Mat_alloc(1, 10); // FC layer weights (flatten to 10 outputs)
    Mat bias = Mat_alloc(1, 10);
    Mat conv_out = Mat_alloc(26, 26);
    Mat pooled = Mat_alloc(13, 13);
    Mat flat = Mat_alloc(1, 13*13);
    Mat out = Mat_alloc(1, 10);

    Mat_rand(kernel, -1, 1);
    Mat_rand(weights, -1, 1);
    Mat_fill(bias, 0);

    float rate = 0.01f;

    for (int epoch = 0; epoch < 1; epoch++)
    {
        float loss = 0;

        for (int i = 0; i < count; i++)
        {
            Mat input = (Mat) {
                .rows = 28,
                .cols = 28,
                .stride = 28,
                .es = (float*) malloc(28*28*sizeof(float))
            };

            for (int j = 0; j < 28*28; j++)
                input.es[j] = images[i * 28 * 28 + j] / 255.0f;

            Mat target = Mat_alloc(1, 10);
            Mat_fill(target, 0);
            MAT_AT(target, 0, labels[i]) = 1.0f;

            Conv2D(conv_out, input, kernel, 1, 0);
            MaxPool2D(pooled, conv_out, 2, 2, 2);
            Flatten(flat, &pooled, 1, 13, 13);

            Mat_dot(out, flat, weights);
            Mat_sum(out, bias);
            Softmax(out);

            for (int k = 0; k < 10; k++)
            {
                float d = MAT_AT(out, 0, k) - MAT_AT(target, 0, k);
                loss += d * d;

                for (int j = 0; j < flat.cols; j++)
                    MAT_AT(weights, 0, k) -= rate * d * MAT_AT(flat, 0, j);
                
                MAT_AT(bias, 0, k) -= rate * d;
            }

            free(input.es);
            Mat_free(target);
        }

        printf("Epoch %d loss: %f\n", epoch, loss / count);
    }

    FILE* f = fopen("model.cnn", "wb");
    Mat_save(f, kernel);
    Mat_save(f, weights);
    Mat_save(f, bias);
    fclose(f);

    free(images);
    free(labels);
    Mat_free(kernel);
    Mat_free(weights);
    Mat_free(bias);
    Mat_free(conv_out);
    Mat_free(pooled);
    Mat_free(flat);
    Mat_free(out);

    return EXIT_SUCCESS;
}