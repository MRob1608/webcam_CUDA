#include <stdio.h>
#include <stdlib.h>

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
/*

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

        // Clamping
        r0 = r0 < 0 ? 0 : (r0 > 255 ? 255 : r0);
        g0 = g0 < 0 ? 0 : (g0 > 255 ? 255 : g0);
        b0 = b0 < 0 ? 0 : (b0 > 255 ? 255 : b0);
        r1 = r1 < 0 ? 0 : (r1 > 255 ? 255 : r1);
        g1 = g1 < 0 ? 0 : (g1 > 255 ? 255 : g1);
        b1 = b1 < 0 ? 0 : (b1 > 255 ? 255 : b1);

        // BGR instead of RGB
        rgb[j]     = b0;
        rgb[j + 1] = g0;
        rgb[j + 2] = r0;
        rgb[j + 3] = b1;
        rgb[j + 4] = g1;
        rgb[j + 5] = r1;
    }
}

*/