#include <stdio.h>
#include <stdlib.h>


__global__ void yuyv_to_bgr_CUDA(unsigned char *yuyv, unsigned char *rgb, int height, int width);

__global__ void mirror_image_kernel(unsigned char* rgb, unsigned char* mirrored_rgb, int height, int width);

void mirror_image_gpu(unsigned char* rgb, int width, int height);