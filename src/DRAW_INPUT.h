#pragma once

#define WIN32_LEAN_AND_MEAN

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

uint16_t SCREEN_WIDTH, SCREEN_HEIGHT,
         SCREEN_STANDARD,
         CANVAS_SIZE, CANVAS_X, CANVAS_Y,
         CELL_LEN, CELL_SIZE;

void SetScreenConstants(int screen_width, int screen_height)
{
    SCREEN_WIDTH  = (uint16_t)screen_width;
    SCREEN_HEIGHT = (uint16_t)screen_height;

    SCREEN_STANDARD = SCREEN_WIDTH > SCREEN_HEIGHT ? SCREEN_HEIGHT : SCREEN_WIDTH;

    CANVAS_SIZE = SCREEN_STANDARD / 2;

    CANVAS_X = (SCREEN_WIDTH - CANVAS_SIZE * 3) / 2;
    CANVAS_Y = (SCREEN_HEIGHT - CANVAS_SIZE) / 2;

    CELL_LEN = 28;
    CELL_SIZE = CANVAS_SIZE / CELL_LEN;
}

inline void ClearCanvas(uint8_t* data)
{
    for (uint16_t i = 0; i < CELL_LEN * CELL_LEN; i++)
        data[i] = 0;
}

void DrawInCanvas(HDC hdc, uint8_t* data)
{
    static uint32_t* dib = NULL;
    static int dib_size = 0;

    if (CANVAS_SIZE <= 0) return;

    if (dib_size != CANVAS_SIZE)
    {
        free(dib);

        dib = (uint32_t*) malloc(sizeof(uint32_t) * CANVAS_SIZE * CANVAS_SIZE);
        dib_size = CANVAS_SIZE;
    }

    if (dib == NULL)
        return;

    for (int py = 0; py < CANVAS_SIZE; py++)
    {
        int cell_y = py / CELL_SIZE;
        if (cell_y >= CELL_LEN) cell_y = CELL_LEN - 1;

        for (int px = 0; px < CANVAS_SIZE; px++)
        {
            int cell_x = px / CELL_SIZE;
            if (cell_x >= CELL_LEN) cell_x = CELL_LEN - 1;

            uint8_t v = data[cell_y * CELL_LEN + cell_x];
            uint32_t col = 0xFF000000 | (v << 16) | (v << 8) | v;

            dib[py * CANVAS_SIZE + px] = col;
        }
    }

    BITMAPINFO bmi;
    ZeroMemory(&bmi, sizeof(bmi));
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = CANVAS_SIZE;
    bmi.bmiHeader.biHeight = -CANVAS_SIZE;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    if (dib != NULL)
    {
        SetDIBitsToDevice(hdc,
            0, 0, 
            CANVAS_SIZE, CANVAS_SIZE, 
            0, 0, 
            0, CANVAS_SIZE, 
            dib,
            &bmi,
            DIB_RGB_COLORS);
    }
}

void DrawCircleInCanvas(uint8_t* data, int x, int y)
{
    if (x < CANVAS_X || x >= CANVAS_X + CANVAS_SIZE ||
        y < CANVAS_Y || y >= CANVAS_Y + CANVAS_SIZE)
        return;

    int cell_x = (x - CANVAS_X) / CELL_SIZE;
    int cell_y = (y - CANVAS_Y) / CELL_SIZE;

    const int radius = 2;

    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
            int cx = cell_x + dx;
            int cy = cell_y + dy;

            if (cx < 0 || cx >= CELL_LEN || cy < 0 || cy >= CELL_LEN)
                continue;

            float dist = sqrtf((float)(dx * dx + dy * dy));
            if (dist > radius)
                continue;

            float intensity = 1.0f - (dist / radius);
            if (intensity < 0.0f) intensity = 0.0f;

            int idx = cy * CELL_LEN + cx;
            int v = data[idx] + (int)(intensity * 64);

            if (v > 255) v = 255;
            data[idx] = (uint8_t)v;
        }
    }
}