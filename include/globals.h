// File: globals.h
#ifndef GLOBALS_H
#define GLOBALS_H

extern int GPU;
extern int EDGE_DET;
extern int OPTICAL;
extern int BILINEAR;

extern int prev_x;
extern int prev_y;
extern unsigned char* dev_rgb;
extern unsigned char* dev_gray;
extern unsigned char* device_rgb, *device_gray, *device_output, *device_yuyv, *device_blur, *device_sharpened_rgb;

extern unsigned char* device_prev_rgb, *device_prev_gray, *device_prev_blur, *device_scaled_rgb;

extern float* d_Ix, *d_It, *d_Iy, *d_u, *d_v, *d_u_avg, *d_v_avg, *mag;

#endif // GLOBALS_H

