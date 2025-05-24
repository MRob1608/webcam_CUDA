#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "image_manipulation.cuh"

__device__ int idx(int x, int y, int width) {
    return 4 * (y * width + x);
}

__global__ void gray_scale_conversion(unsigned char* rgb, unsigned char* gray, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= height * width) return;

    int pixel_idx = 4*tid;

    gray[tid] = rgb[pixel_idx] * 0.114f + rgb[pixel_idx + 1] * 0.587f + rgb[pixel_idx + 2] * 0.299f;

}


__global__ void edge_detection_overlay(unsigned char* rgb, unsigned char* output, unsigned char* grayscale, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)  //escludo i pixel ai bordi
        return;

    float Gx = 0.0f, Gy = 0.0f;
    int thold = 40;

    const int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    const int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int ni = x + i;
            int nj = y + j;
            int index = idx(ni, nj, width);

            float gray = grayscale[nj * width + ni];


            Gx += gray * sobel_x[j+1][i+1];
            Gy += gray * sobel_y[j+1][i+1];
        }
    }

    float magnitude = sqrtf(Gx*Gx + Gy*Gy);
    if (magnitude > 255) magnitude = 255;

    int pixel_idx = idx(x,y,width);

    if (magnitude > thold) {
        output[pixel_idx] = rgb[pixel_idx];
        output[pixel_idx + 1] = rgb[pixel_idx+1];
        output[pixel_idx + 2] = 255;
        output[pixel_idx + 3] = 0;
    } else {
        output[pixel_idx]     = rgb[pixel_idx];
        output[pixel_idx + 1] = rgb[pixel_idx+1];
        output[pixel_idx + 2] = rgb[pixel_idx+2];
        output[pixel_idx + 3] = 0;
    }
}