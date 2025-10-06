#include <stdint.h>
#include <time.h>

#define NN_IMPLEMENTATION
#include "NN.h"

#include "raylib.h"

#define IMG_FACTOR 100
#define IMG_WIDTH  (IMG_FACTOR * 16)
#define IMG_HEIGHT (IMG_FACTOR * 9)

uint32_t img_pixels[IMG_WIDTH * IMG_HEIGHT];

void NN_render_raylib(NN nn)
{
    Color background_color = { 0x18, 0x18, 0x18, 0xFF };

    Color low_color  = { 0xFF, 0x18, 0xFF, 0xFF };
    Color high_color = { 0x00, 0xFF, 0x00, 0xFF };

    ClearBackground(background_color);

    int neuron_radius = 25;

    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    
    int nn_width = IMG_WIDTH - layer_border_hpad * 2;
    int nn_height = IMG_HEIGHT - layer_border_vpad * 2;
    
    int nn_x = IMG_WIDTH / 2 - nn_width / 2;
    int nn_y = IMG_HEIGHT / 2 - nn_height / 2;

    size_t arch_count = nn.count + 1;
    int layer_hpad = nn_width / arch_count;

    for (size_t iv = 0; iv < arch_count; iv++)
    {
        int layer_vpad1 = nn_height / nn.as[iv].cols;

        for (size_t i = 0; i < nn.as[iv].cols; i++)
        {
            int cx1 = nn_x + layer_hpad * iv + layer_hpad / 2;
            int cy1 = nn_y + layer_vpad1 * i + layer_vpad1 / 2;
            
            if (iv + 1 < arch_count)
            {
                int layer_vpad2 = nn_height / nn.as[iv + 1].cols;
                
                for (size_t ii = 0; ii < nn.as[iv + 1].cols; ii++)
                {
                    /* 
                     * i  - rows of ws
                     * ii - cols of ws
                     */

                    int cx2 = nn_x + layer_hpad * (iv + 1) + layer_hpad / 2;
                    int cy2 = nn_y + layer_vpad2 * ii + layer_vpad2 / 2;

                    float value = sigmoidf(MAT_AT(nn.ws[iv], i, ii));
                    high_color.a = floorf(value * 255.0F);
                    
                    float thick = value * 1.0F + 1.5F;

                    Vector2 start = { cx1, cy1 };
                    Vector2 end   = { cx2, cy2 };

                    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }

            if (iv > 0)
            {
                high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[iv - 1], 0, i)));

                DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
            }
            else
            {
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}

#define TRAIN_COUNT (size_t) 2E4

#define RATE (float) 1E-0

#define BITS 4

int main(void)
{
    size_t n = (1 << BITS);
    size_t rows = n * n;

    Mat ti = Mat_alloc(rows, BITS * 2);
    Mat to = Mat_alloc(rows, BITS + 1);

    for (size_t i = 0; i < ti.rows; i++)
    {
        size_t x = i / n;
        size_t y = i % n;

        size_t z = x + y;

        for (size_t ii = 0; ii < BITS; ii++)
        {
            MAT_AT(ti, i, ii)        = (x >> ii) & 1;
            MAT_AT(ti, i, ii + BITS) = (y >> ii) & 1;
            MAT_AT(to, i, ii) = (z >> ii) & 1;
        }

        MAT_AT(to, i, BITS) = z >= n;
    }

    size_t arch[] = { BITS * 2, BITS * 4, BITS * 2, BITS + 1 };

    NN nn  = NN_alloc(arch, ARRAY_LEN(arch));
    NN gnn = NN_alloc(arch, ARRAY_LEN(arch));

    NN_rand(nn, 0, 1);

    InitWindow(IMG_WIDTH, IMG_HEIGHT, "adder");

    SetTargetFPS(1000);

    size_t i = 0;

    while (!WindowShouldClose())
    {
        if (i < TRAIN_COUNT)
        {
            NN_backprop(nn, gnn, ti, to);
            NN_learn(nn, gnn, RATE);

            i++;

            if (i % 1000 == 0) printf("%zu : cost = %f\n", i, NN_cost(nn, ti, to));
        }

        BeginDrawing();

        NN_render_raylib(nn);

        EndDrawing();
    }

    size_t fails = 0;

    for (size_t x = 0; x < n; x++)
        for (size_t y = 0; y < n; y++)
        {
            size_t z = x + y;

            for (size_t ii = 0; ii < BITS; ii++)
            {
                MAT_AT(NN_INPUT(nn), 0, ii)        = (x >> ii) & 1;
                MAT_AT(NN_INPUT(nn), 0, ii + BITS) = (y >> ii) & 1;
            }
            
            NN_forward(nn);
            
            if (MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5F)
            {
                if (z < n)
                {
                    printf("%zu + %zu = (OVERFLOW <> %zu)\n", x, y, z);

                    fails++;
                }
            }
            else
            {
                size_t a = 0;

                for(size_t ii = 0; ii < BITS; ii++)
                {
                    size_t bit = MAT_AT(NN_OUTPUT(nn), 0, ii) > 0.5F;

                    a |= bit << ii;
                }

                if (z != a)
                {
                    printf("%zu + %zu = (%zu <> %zu)\n", x, y, z, a);

                    fails++;
                }
            }
        }

    if (fails == 0)
    {
        printf("\nOK");
    }

    Mat_free(ti);
    Mat_free(to);

    NN_free(nn);
    NN_free(gnn);

    return EXIT_SUCCESS;
}