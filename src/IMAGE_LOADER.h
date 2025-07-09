#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Load PGM image to float array (normalized 0.0 to 1.0)
void load_pgm(const char* filename, float* out, size_t size);

// Load IDX image and label files
unsigned char* load_idx_images(const char* filename, int* count, int* rows, int* cols);
unsigned char* load_idx_labels(const char* filename, int* count);

void load_pgm(const char* filename, float* out, size_t size)
{
    FILE* f = fopen(filename, "rb");

    if (!f)
    {
        fprintf(stderr, "Error: Cannot open PGM file %s\n", filename);

        exit(EXIT_FAILURE);
    }

    char magic[3];
    int width, height, maxval;
    fscanf(f, "%2s\n%d %d\n%d\n", magic, &width, &height, &maxval);

    if (strcmp(magic, "P5") != 0 || width * height != size)
    {
        fprintf(stderr, "Invalid or unsupported PGM file\n");

        exit(EXIT_FAILURE);
    }

    unsigned char* pixels = (unsigned char*) malloc(size);
    fread(pixels, 1, size, f);
    fclose(f);

    for (size_t i = 0; i < size; i++)
        out[i] = pixels[i] / 255.0f;

    free(pixels);
}

unsigned char* load_idx_images(const char* filename, int* count, int* rows, int* cols)
{
    FILE* f = fopen(filename, "rb");

    if (!f)
        return NULL;
    
    uint8_t header[16];
    fread(header, 1, 16, f);
    
    *count = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | header[7];
    *rows  = (header[8] << 24) | (header[9] << 16) | (header[10] << 8) | header[11];
    *cols  = (header[12] << 24) | (header[13] << 16) | (header[14] << 8) | header[15];

    unsigned char* data = (unsigned char*) malloc((*count) * (*rows) * (*cols));
    fread(data, 1, (*count)*(*rows)*(*cols), f);
    fclose(f);

    return data;
}

unsigned char* load_idx_labels(const char* filename, int* count)
{
    FILE* f = fopen(filename, "rb");

    if (!f)
        return NULL;
    
    uint8_t header[8];
    fread(header, 1, 8, f);

    *count = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | header[7];

    unsigned char* labels = (unsigned char*) malloc(*count);
    fread(labels, 1, *count, f);
    fclose(f);

    return labels;
}

#endif // IMAGE_LOADER_H