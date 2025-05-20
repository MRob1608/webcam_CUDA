#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>

void init_x11(int width, int height);

void display_frame(unsigned char* rgb_data, int width, int height);