#pragma once

#include <stdio.h>
#include <stdlib.h>

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>

SDL_Window *window_global = NULL;
SDL_Renderer *renderer_global = NULL;
SDL_Texture *texture = NULL;

void init_SDL(void)
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        
        exit(EXIT_FAILURE);
    }

    window_global = SDL_CreateWindow("Draw Digit", 280, 280, SDL_WINDOW_RESIZABLE);
    
    if (!window_global)
    {
        fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());

        SDL_Quit();

        exit(EXIT_FAILURE);
    }

    renderer_global = SDL_CreateRenderer(window_global, NULL);
    
    if (!renderer_global)
    {
        fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError());

        SDL_DestroyWindow(window_global);
        SDL_Quit();

        exit(EXIT_FAILURE);
    }

    texture = SDL_CreateTexture(renderer_global, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, 280, 280);
    
    if (!texture)
    {
        fprintf(stderr, "SDL_CreateTexture Error: %s\n", SDL_GetError());

        SDL_DestroyRenderer(renderer_global);
        SDL_DestroyWindow(window_global);
        SDL_Quit();

        exit(EXIT_FAILURE);
    }

    SDL_SetRenderTarget(renderer_global, texture);
    SDL_SetRenderDrawColor(renderer_global, 255, 255, 255, 255);
    SDL_RenderClear(renderer_global);
    SDL_SetRenderTarget(renderer_global, NULL);
}

int process_events(int *quit)
{
    SDL_Event event;
    int recognized = 0;

    while (SDL_PollEvent(&event))
        switch (event.type)
        {
            case SDL_EVENT_QUIT:
                *quit = 1;
                break;

            case SDL_EVENT_KEY_DOWN:
                if (event.key.key == SDLK_ESCAPE)
                    *quit = 1;
                
                else if (event.key.key == SDLK_RETURN || event.key.key == SDLK_KP_ENTER)
                    recognized = 1;
                
                else if (event.key.key == SDLK_C)
                {
                    SDL_SetRenderTarget(renderer_global, texture);
                    SDL_SetRenderDrawColor(renderer_global, 255, 255, 255, 255);
                    SDL_RenderClear(renderer_global);
                    SDL_SetRenderTarget(renderer_global, NULL);
                }

                break;

            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_MOTION:
            {
                Uint32 mouseState = SDL_GetMouseState(NULL, NULL);

                if (mouseState & SDL_BUTTON_LMASK)
                {
                    int x = event.motion.x;
                    int y = event.motion.y;

                    SDL_SetRenderTarget(renderer_global, texture);
                    SDL_SetRenderDrawColor(renderer_global, 0, 0, 0, 255);
                    SDL_FRect rect = { x - 24.0f, y - 24.0f, 48.0f, 48.0f };
                    SDL_RenderFillRect(renderer_global, &rect);
                    SDL_SetRenderTarget(renderer_global, NULL);
                }

                break;
            }
        }

    return recognized;
}

float* get_drawn_digit(SDL_Renderer* renderer)
{
    const int full_size = 280;
    const int small_size = 28;

    SDL_Rect read_rect = { 0, 0, full_size, full_size };

    SDL_Surface* surface = SDL_RenderReadPixels(renderer, &read_rect);

    if (!surface)
    {
        fprintf(stderr, "SDL_RenderReadPixels failed: %s\n", SDL_GetError());

        return NULL;
    }

    Uint32* pixels = (Uint32*) surface->pixels;
    float* digit_data = (float*) malloc(sizeof(float) * small_size * small_size);

    if (!digit_data)
    {
        fprintf(stderr, "Failed to allocate digit data buffer\n");

        SDL_DestroySurface(surface);

        return NULL;
    }

    // Downsample from 280x280 → 28x28
    for (int y = 0; y < small_size; y++)
        for (int x = 0; x < small_size; x++)
        {
            int sum = 0;

            for (int dy = 0; dy < 10; dy++)
                for (int dx = 0; dx < 10; dx++)
                {
                    int src_x = x * 10 + dx;
                    int src_y = y * 10 + dy;
                    Uint32 pixel = pixels[src_y * full_size + src_x];

                    Uint8 r, g, b;
                    const SDL_PixelFormatDetails* fmt_details = SDL_GetPixelFormatDetails(surface->format);
                    SDL_GetRGB(pixel, fmt_details, NULL, &r, &g, &b);
                    
                    int luminance = (int) (0.299f * r + 0.587f * g + 0.114f * b);
                    sum += luminance;
                }

            float avg = sum / 100.0f;
            digit_data[y * small_size + x] = 1.0f - (avg / 255.0f); // Black:1.0, White:0.0
        }

    SDL_DestroySurface(surface);
    
    return digit_data;
}

void free_drawn_digit(float* digit_data)
{
    if (digit_data)
        free(digit_data);
}

void cleanup_SDL(void)
{
    if (texture)
        SDL_DestroyTexture(texture);

    if (renderer_global)
        SDL_DestroyRenderer(renderer_global);

    if (window_global)
        SDL_DestroyWindow(window_global);

    SDL_Quit();
}