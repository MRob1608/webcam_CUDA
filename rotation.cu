#include <stdio.h>
#include <stdlib.h>
#include "globals.h"
#include "capture_camera.h"
#include <math.h>


__global__ void rotate_90(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int src_idx = (y * width + x) * 4;
        int dst_idx = ((x) * height + (height - 1 - y)) * 4;

        output[dst_idx + 0] = input[src_idx + 0];
        output[dst_idx + 1] = input[src_idx + 1];
        output[dst_idx + 2] = input[src_idx + 2];
        output[dst_idx + 3] = input[src_idx + 3];
    }
}

__global__ void rotate_180(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int src_idx = (y * width + x) * 4;
        int dst_idx = ((height - 1 - y) * width + (width - 1 - x)) * 4;

        output[dst_idx + 0] = input[src_idx + 0];
        output[dst_idx + 1] = input[src_idx + 1];
        output[dst_idx + 2] = input[src_idx + 2];
        output[dst_idx + 3] = input[src_idx + 3];
    }
}


__global__ void rotate_270(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int src_idx = (y * width + x) * 4;
        int dst_idx = ((width - 1 - x) * height + y) * 4;

        output[dst_idx + 0] = input[src_idx + 0];
        output[dst_idx + 1] = input[src_idx + 1];
        output[dst_idx + 2] = input[src_idx + 2];
        output[dst_idx + 3] = input[src_idx + 3];
    }
}

void rotate_image(unsigned char* rgb, int width, int height, int degrees) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaMemcpy(device_rgb,(unsigned char*)rgb, width * height * 4, cudaMemcpyHostToDevice);

    switch (degrees) {
        case 90:
            rotate_90<<<grid, block>>>(device_rgb, device_rotated_rgb, width, height);
            break;
        case 180:
            rotate_180<<<grid, block>>>(device_rgb, device_rotated_rgb, width, height);
            break;
        case 270:
            rotate_270<<<grid, block>>>(device_rgb, device_rotated_rgb, width, height);
            break;
        default:
            return;
    }
    cudaMemcpy((unsigned char*)rgb, device_rotated_rgb, width * height *4, cudaMemcpyDeviceToHost);
}


void alloc_Rotation(camera_t* camera) {
    if (!GPU) {
      cudaMalloc(&device_rgb, camera->width * camera->height * 4);
    }
    cudaMalloc(&device_rotated_rgb, camera->width * camera->height * 4);
}

void free_Rotation(void) {
    if (!GPU) {
      cudaFree(device_rgb);
    }
    cudaFree(device_rotated_rgb);
}