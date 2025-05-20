#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <X11/Xutil.h>

Display *display;
Window window;
GC gc;
XImage *ximage;

void init_x11(int width, int height) {
    display = XOpenDisplay(NULL);
    int screen = DefaultScreen(display);
    window = XCreateSimpleWindow(display, RootWindow(display, screen),
                                  10, 10, width, height, 1,
                                  BlackPixel(display, screen),
                                  WhitePixel(display, screen));

    XMapWindow(display, window);
    gc = DefaultGC(display, screen);

    XFlush(display);
}

void display_frame(unsigned char* rgb_data, int width, int height) {
    int bytes_per_pixel = 3;
    int bytes_per_line = width * bytes_per_line;
    if (!ximage) {
        ximage = XCreateImage(display, DefaultVisual(display, 0), 24,
                              ZPixmap, 0, (char*)rgb_data, width, height, 32, bytes_per_line);
    } else {
        ximage->data = (char*)rgb_data;
    }

    XPutImage(display, window, gc, ximage, 0, 0, 0, 0, width, height);
    XFlush(display);
    

}


