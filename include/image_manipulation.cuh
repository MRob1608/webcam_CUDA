#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "capture_camera.h"


__device__ int idx(int x, int y, int width);

__global__ void gray_scale_conversion(unsigned char* rgb, unsigned char* gray, int width, int height);

__global__ void edge_detection_overlay(unsigned char* rgb, unsigned char* output, unsigned char* grayscale, int width, int height);

__global__ void compute_derivatives(const unsigned char* prev_gray,const unsigned char* curr_gray,float* Ix, float* Iy, float* It,int width, int height);

__global__ void average_uv(const float* u_in, const float* v_in,float* u_avg, float* v_avg,int width, int height);

__global__ void update_uv(const float* Ix, const float* Iy, const float* It,const float* u_avg, const float* v_avg,float* u, float* v,float alpha,int width, int height);

__global__ void compute_flow_magnitude(const float* u, const float* v,float* mag,int width, int height);

__global__ void blur_image(unsigned char* gray, unsigned char* blur, int width, int height);

void draw_square(unsigned char* image, int width, int height, int center_x, int center_y,int size);

__global__ void scale_image_cn_kernel(const unsigned char* input, int src_width, int src_height, unsigned char* output, int dst_width, int dst_height);

__global__ void scale_image_bilinear_kernel(unsigned char* input, unsigned char* output, int in_width, int in_height, int out_width, int out_height);

__global__ void image_sharpen(unsigned char* input, unsigned char* output, int width, int height);
