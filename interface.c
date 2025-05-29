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

    Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wm_delete_window, 1);

    XSelectInput(display, window, ExposureMask | KeyPressMask | StructureNotifyMask);

    XFlush(display);
}

void display_frame(unsigned char* rgb_data, int width, int height) {
    int bytes_per_pixel = 4;
    int bytes_per_line = width * bytes_per_pixel; // allineato a 4 byte
    if (ximage) {
        XDestroyImage(ximage); // importante: evita memory leak
    }
    int depth = DefaultDepth(display, DefaultScreen(display));

    unsigned char* rgb_copy = malloc(height * width * 4);
    memcpy(rgb_copy, rgb_data, height * width * 4);


    ximage = XCreateImage(display,
                          DefaultVisual(display, 0),
                          24,                 // depth (bit per pixel)
                          ZPixmap,
                          0,
                          (char*)rgb_copy,
                          width,
                          height,
                          8,                  // bitmap_pad in bit
                          bytes_per_line);    // bytes per line, esplicito!
    if (!ximage) {
        fprintf(stderr, "XCreateImage failed! width=%d height=%d depth=%d bytes_per_line=%d\n",
                width, height, depth, bytes_per_line);
        free(rgb_copy);
        return;
    }
    XPutImage(display, window, gc, ximage, 0, 0, 0, 0, width, height);
    XFlush(display);
}


