#include <stdio.h>
#include <stdlib.h>
#include "capture_camera.h"

void rotate_image(unsigned char* rgb, int width, int height, int degrees);

__global__ void rotate_90(unsigned char* input, unsigned char* output, int width, int height);

__global__ void rotate_180(unsigned char* input, unsigned char* output, int width, int height);

__global__ void rotate_270(unsigned char* input, unsigned char* output, int width, int height);

void alloc_Rotation(camera_t* camera);

void free_Rotation(void);