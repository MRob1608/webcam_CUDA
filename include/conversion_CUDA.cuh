#include <stdio.h>
#include <stdlib.h>


__global__ void yuyv_to_bgr_CUDA(unsigned char *yuyv, unsigned char *rgb, int height, int width);