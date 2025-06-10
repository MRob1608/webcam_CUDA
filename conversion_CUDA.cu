#include <stdio.h>
#include <stdlib.h>
#include "conversion_CUDA.cuh"
#include "globals.h"

// Kernel to convert an image from YUYV format to BGRA
__global__ void yuyv_to_bgr_CUDA(unsigned char *yuyv, unsigned char *rgb, int height, int width)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= height * width / 2) {
        return;
    }

    int y, v, u;
    float r, g, b;

    y = yuyv[4*tid + 0]; //y0
    u = yuyv[4*tid + 1]; //u0
    v = yuyv[4*tid + 3]; //v0

    r = y + 1.4065 * (v - 128);			     //r0
    g = y - 0.3455 * (u - 128) - 0.7169 * (v - 128); //g0
    b = y + 1.1790 * (u - 128);			     //b0

    if (r < 0)
	r = 0;
    else if (r > 255)
	r = 255;
    if (g < 0)
	g = 0;
    else if (g > 255)
	g = 255;
    if (b < 0)
	b = 0;
    else if (b > 255)
	b = 255;

    rgb[8*tid + 0] = (unsigned char)b;
    rgb[8*tid + 1] = (unsigned char)g;
    rgb[8*tid + 2] = (unsigned char)r;
    rgb[8*tid + 3] = (unsigned char)0;


    //second pixel
    u = yuyv[4*tid + 1]; //u0
    y = yuyv[4*tid + 2]; //y1
    v = yuyv[4*tid + 3]; //v0

    r = y + 1.4065 * (v - 128);			     //r1
    g = y - 0.3455 * (u - 128) - 0.7169 * (v - 128); //g1
    b = y + 1.1790 * (u - 128);			     //b1

    if (r < 0)
	r = 0;
    else if (r > 255)
	r = 255;
    if (g < 0)
	g = 0;
    else if (g > 255)
	g = 255;
    if (b < 0)
	b = 0;
    else if (b > 255)
	b = 255;

    rgb[8*tid + 4] = (unsigned char)b;
    rgb[8*tid + 5] = (unsigned char)g;
    rgb[8*tid + 6] = (unsigned char)r;
    rgb[8*tid + 7] = (unsigned char)0;
}

//Kernel to mirror the image
__global__ void mirror_image_kernel(unsigned char* rgb, unsigned char* mirrored_rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int src_idx = ((width * y) + x) * 4;
    int dest_idx = ((width * y) + width - x - 1) * 4;   //position of the new pixel


    mirrored_rgb[dest_idx] = rgb[src_idx];
    mirrored_rgb[dest_idx+1] = rgb[src_idx+1];
    mirrored_rgb[dest_idx+2] = rgb[src_idx+2];
    mirrored_rgb[dest_idx+3] = rgb[src_idx+3];
}

//Wrapper for the mirroring kernel
void mirror_image_gpu(unsigned char* rgb, int width, int height) {
    dim3 blockSize(16, 16);  
    dim3 gridSize(
      (width + blockSize.x - 1) / blockSize.x,   //ceil of width / blocksize
      (height + blockSize.y - 1) / blockSize.y   //ceil of height / blocksize
    );

    cudaMemcpy(device_rgb, rgb, width * height * 4, cudaMemcpyHostToDevice);

    mirror_image_kernel<<<gridSize, blockSize>>>(device_rgb, device_mirrored_rgb, width, height);

    cudaMemcpy(rgb, device_mirrored_rgb, width * height * 4, cudaMemcpyDeviceToHost);
}
