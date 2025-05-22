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
/*
void display_frame(unsigned char* rgb_data, int width, int height) {
    int bytes_per_pixel = 3;
    int bytes_per_line = width * 3;
    if (!ximage) {
        ximage = XCreateImage(display, DefaultVisual(display, 0), 24,
                              ZPixmap, 0, (char*)rgb_data, width, height, 32, 0);
    } else {
        ximage->data = (char*)rgb_data;
    }

    XPutImage(display, window, gc, ximage, 0, 0, 0, 0, width, height);
    XFlush(display);
    

}
*/

void display_frame(unsigned char* rgb_data, int width, int height) {
    int bytes_per_pixel = 4;
    int bytes_per_line = width * bytes_per_pixel; // allineato a 4 byte
    if (ximage) {
        XDestroyImage(ximage); // importante: evita memory leak
    }
    int depth = DefaultDepth(display, DefaultScreen(display));

    ximage = XCreateImage(display,
                          DefaultVisual(display, 0),
                          24,                 // depth (bit per pixel)
                          ZPixmap,
                          0,
                          (char*)rgb_data,
                          width,
                          height,
                          8,                  // bitmap_pad in bit
                          bytes_per_line);    // bytes per line, esplicito!
    if (!ximage) {
        fprintf(stderr, "XCreateImage failed! width=%d height=%d depth=%d bytes_per_line=%d\n",
                width, height, depth, bytes_per_line);
        return;
    }
    XPutImage(display, window, gc, ximage, 0, 0, 0, 0, width, height);
    XFlush(display);
}


