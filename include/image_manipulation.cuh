#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__device__ int idx(int x, int y, int width);

__global__ void gray_scale_conversion(unsigned char* rgb, unsigned char* gray, int width, int height);

__global__ void edge_detection_overlay(unsigned char* rgb, unsigned char* output, unsigned char* grayscale, int width, int height);