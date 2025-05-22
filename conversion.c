#include <stdio.h>
#include <stdlib.h>
#include "conversion.h"

void yuyv_to_rgb24(unsigned char* yuyv, unsigned char* rgb, int width, int height) {
    int i, j;
    for (i = 0, j = 0; i < width * height * 2; i += 4, j += 6) {
        int y0 = yuyv[i];
        int u  = yuyv[i + 1] - 128;
        int y1 = yuyv[i + 2];
        int v  = yuyv[i + 3] - 128;

        int r0 = y0 + 1.402 * v;
        int g0 = y0 - 0.344136 * u - 0.714136 * v;
        int b0 = y0 + 1.772 * u;

        int r1 = y1 + 1.402 * v;
        int g1 = y1 - 0.344136 * u - 0.714136 * v;
        int b1 = y1 + 1.772 * u;

        rgb[j]     = r0 < 0 ? 0 : r0 > 255 ? 255 : r0;
        rgb[j + 1] = g0 < 0 ? 0 : g0 > 255 ? 255 : g0;
        rgb[j + 2] = b0 < 0 ? 0 : b0 > 255 ? 255 : b0;

        rgb[j + 3] = r1 < 0 ? 0 : r1 > 255 ? 255 : r1;
        rgb[j + 4] = g1 < 0 ? 0 : g1 > 255 ? 255 : g1;
        rgb[j + 5] = b1 < 0 ? 0 : b1 > 255 ? 255 : b1;
    }
}


void yuyv_to_bgr_pixel(unsigned char *yuyv, unsigned char *rgb)
{
    int y, v, u;
    float r, g, b;

    y = yuyv[0]; //y0
    u = yuyv[1]; //u0
    v = yuyv[3]; //v0

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

    rgb[0] = (unsigned char)b;
    rgb[1] = (unsigned char)g;
    rgb[2] = (unsigned char)r;
    rgb[3] = (unsigned char)0;

    //second pixel
    u = yuyv[1]; //u0
    y = yuyv[2]; //y1
    v = yuyv[3]; //v0

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

    rgb[4] = (unsigned char)b;
    rgb[5] = (unsigned char)g;
    rgb[6] = (unsigned char)r;
    rgb[7] = (unsigned char)0;
}


int yuyv_to_bgr(unsigned char *yuyv, unsigned char *rgb, int height, int width)
{
    unsigned char temp = 0;
    long yuv_size = height * width * 2;
    long rgb_size = height * width * 4;

    if (yuyv == NULL || rgb == NULL)
	return 0;

    for (int i = 0, j = 0; i < rgb_size && j < yuv_size; i += 8, j += 4)
    {
	yuyv_to_bgr_pixel(&yuyv[j], &rgb[i]);
    }
    return 1;
}