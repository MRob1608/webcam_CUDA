// File: globals.c (o main.cu)
#include "globals.h"


int GPU = 1;              // if true, the conversion from yuyv to bgr2 is calculated with the GPU
int EDGE_DET = 1;         // if true, the edge detection filter will be applied
int OPTICAL = 0;          // if true, the optical flow filter will be applied
int BILINEAR = 0;         // if true, the scaling is done with the bilinear technique else, the closest neighbor is used
int ROTATION_ANGLE = 0;   // set the roation angle wich will be used to rotate the image in output (possible values: 0, 90, 180, 270)

unsigned char* device_rgb, *device_gray, *device_output, *device_yuyv, *device_blur, *device_mirrored_rgb;

unsigned char* device_prev_rgb, *device_prev_gray, *device_prev_blur, *device_scaled_rgb, *device_sharpened_rgb, *device_rotated_rgb;

float* d_Ix, *d_It, *d_Iy, *d_u, *d_v, *d_u_avg, *d_v_avg, *mag;

int prev_x = 0, prev_y = 0; //previous coordinates of the center of the square in optical flow
