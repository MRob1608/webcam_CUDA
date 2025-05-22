#include <stdio.h>
#include <stdlib.h>


void yuyv_to_rgb24(unsigned char* yuyv, unsigned char* rgb, int width, int height);

int yuyv_to_bgr(unsigned char *yuyv, unsigned char *rgb, int height, int width);

void yuyv_to_bgr_pixel(unsigned char *yuyv, unsigned char *rgb);