#include <stdio.h>
#include <stdlib.h>

#include "NN.h"
#include "DRAW_INPUT.h"

int main()
{
    init_SDL();

    FILE *file = fopen("data/model.nn", "rb");

    if (!file)
    {
        fprintf(stderr, "Failed to open model file\n");

        return EXIT_FAILURE;
    }

    NN nn = NN_load(file);
    fclose(file);

    if (nn.count == 0)
    {
        fprintf(stderr, "Failed to load model\n");

        cleanup_SDL();

        return EXIT_FAILURE;
    }

    printf("Draw digit with mouse. Press ENTER to recognize, C to clear, ESC to quit.\n");

    int quit = 0;

    while (!quit){
        int recognized = process_events(&quit);

        SDL_SetRenderDrawColor(renderer_global, 255, 255, 255, 255);
        SDL_RenderClear(renderer_global);
        
        SDL_RenderTexture(renderer_global, texture, NULL, NULL);
        SDL_RenderPresent(renderer_global);
        
        if (recognized)
        {
            float* input = get_drawn_digit(renderer_global);
            
            if (input)
            {
                for (int i = 0; i < 28 * 28; i++)
                    MAT_AT(NN_INPUT(nn), 0, i) = input[i];

                free(input);
                NN_forward(nn);

                printf("Prediction:\n");

                for (int i = 0; i < 10; i++)
                    printf("Digit %d: %.3f\n", i, MAT_AT(NN_OUTPUT(nn), 0, i));
                
                printf("-----------------\n");
            }
        }
    }

    NN_free(nn);
    cleanup_SDL();

    return EXIT_SUCCESS;
}