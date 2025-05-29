#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "image_manipulation.cuh"

__device__ int idx(int x, int y, int width) {
    return 4 * (y * width + x);
}

__global__ void blur_image(unsigned char* gray, unsigned char* blur, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)  //escludo i pixel ai bordi
        return;

    int pixel_idx = y * width + x;
    int sum = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int p_id = (y+j)*width + x + i;
            sum += gray[p_id];        
        }
    }
    sum /= 9;
    blur[pixel_idx] = sum;
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
    int thold = 50;

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


__global__ void compute_derivatives(
    const unsigned char* prev_gray,
    const unsigned char* curr_gray,
    float* Ix, float* Iy, float* It,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Evita i bordi
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    int idx = y * width + x;

    int idx_left   = y * width + (x - 1);
    int idx_right  = y * width + (x + 1);
    int idx_top    = (y - 1) * width + x;
    int idx_bottom = (y + 1) * width + x;

    // Calcola le medie tra prev e curr
    float avg_center  = 0.5f * (curr_gray[idx] + prev_gray[idx]);
    float avg_left    = 0.5f * (curr_gray[idx_left] + prev_gray[idx_left]);
    float avg_right   = 0.5f * (curr_gray[idx_right] + prev_gray[idx_right]);
    float avg_top     = 0.5f * (curr_gray[idx_top] + prev_gray[idx_top]);
    float avg_bottom  = 0.5f * (curr_gray[idx_bottom] + prev_gray[idx_bottom]);

    // Derivate spaziali (centrali)
    Ix[idx] = (avg_right - avg_left) * 0.5f;
    Iy[idx] = (avg_bottom - avg_top) * 0.5f;

    // Derivata temporale
    It[idx] = (float)(curr_gray[idx]) - (float)(prev_gray[idx]);
}

__global__ void average_uv(
    const float* u_in, const float* v_in,
    float* u_avg, float* v_avg,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Evita i bordi
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    int idx      = y * width + x;
    int idx_left  = y * width + (x - 1);
    int idx_right = y * width + (x + 1);
    int idx_top   = (y - 1) * width + x;
    int idx_bottom= (y + 1) * width + x;

    u_avg[idx] = 0.25f * (u_in[idx_left] + u_in[idx_right] + u_in[idx_top] + u_in[idx_bottom]);
    v_avg[idx] = 0.25f * (v_in[idx_left] + v_in[idx_right] + v_in[idx_top] + v_in[idx_bottom]);
}


__global__ void update_uv(
    const float* Ix, const float* Iy, const float* It,
    const float* u_avg, const float* v_avg,
    float* u, float* v,
    float alpha,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Evita i bordi
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    int idx = y * width + x;

    float ix = Ix[idx];
    float iy = Iy[idx];
    float it = It[idx];

    float ubar = u_avg[idx];
    float vbar = v_avg[idx];

    float denom = alpha * alpha + ix * ix + iy * iy;

    if (denom != 0.0f) {
        float term = (ix * ubar + iy * vbar + it) / denom;
        u[idx] = ubar - ix * term;
        v[idx] = vbar - iy * term;
    }
}

__global__ void compute_flow_magnitude(
    const float* u, const float* v,
    float* mag,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thold = 2;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float dx = u[idx];
    float dy = v[idx];
    float tmp_mag = sqrtf(dx * dx + dy * dy);
    if (tmp_mag < thold) tmp_mag = 0;
    mag[idx] = tmp_mag;
}

