#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <errno.h>

#define WIDTH 1920
#define HEIGHT 1080

// Conversione YUYV (YUV 4:2:2) in RGB24
void yuyv_to_rgb(unsigned char *yuyv, unsigned char *rgb, int width, int height) {
    int i, j;
    for (i = 0, j = 0; i < width * height * 2; i += 4) {
        int y0 = yuyv[i + 0];
        int u  = yuyv[i + 1] - 128;
        int y1 = yuyv[i + 2];
        int v  = yuyv[i + 3] - 128;

        int r, g, b;

        // pixel 1
        r = y0 + 1.402 * v;
        g = y0 - 0.344136 * u - 0.714136 * v;
        b = y0 + 1.772 * u;

        rgb[j++] = r < 0 ? 0 : r > 255 ? 255 : r;
        rgb[j++] = g < 0 ? 0 : g > 255 ? 255 : g;
        rgb[j++] = b < 0 ? 0 : b > 255 ? 255 : b;

        // pixel 2
        r = y1 + 1.402 * v;
        g = y1 - 0.344136 * u - 0.714136 * v;
        b = y1 + 1.772 * u;

        rgb[j++] = r < 0 ? 0 : r > 255 ? 255 : r;
        rgb[j++] = g < 0 ? 0 : g > 255 ? 255 : g;
        rgb[j++] = b < 0 ? 0 : b > 255 ? 255 : b;
    }
}

int main() {
    int fd = open("/dev/video0", O_RDWR);
    if (fd == -1) {
        perror("Errore apertura /dev/video0");
        return 1;
    }

    // Imposta formato YUYV
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("Errore set formato");
        return 1;
    }

    // Richiedi buffer
    struct v4l2_requestbuffers req = {0};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        perror("Errore richiesta buffer");
        return 1;
    }

    // Mappa buffer
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;

    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
        perror("Errore query buffer");
        return 1;
    }

    void* buffer = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (buffer == MAP_FAILED) {
        perror("Errore mmap");
        return 1;
    }

    // Avvia streaming
    int type = buf.type;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("Errore start streaming");
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            perror("Errore QBUF");
            return 1;
        }
        if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            perror("Errore DQBUF");
            return 1;
        }
    }

    // Metti in coda il buffer
    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
        perror("Errore QBUF");
        return 1;
    }
    /*
    // Attendi che il buffer venga riempito con dati validi
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    struct timeval tv = {0};
    tv.tv_sec = 2; // timeout di 2 secondi

    int r = select(fd + 1, &fds, NULL, NULL, &tv);
    if (r == -1) {
        perror("Errore select");
        return 1;
    } else if (r == 0) {
        fprintf(stderr, "Timeout nella cattura del frame\n");
        return 1;
    }
    */
    // Aspetta frame
    if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        perror("Errore DQBUF");
        return 1;
    }

    // Alloca memoria per immagine RGB
    unsigned char *rgb_buffer = malloc(WIDTH * HEIGHT * 3);
    if (!rgb_buffer) {
        perror("Errore malloc RGB");
        return 1;
    }

    // Conversione
    yuyv_to_rgb((unsigned char*)buffer, rgb_buffer, WIDTH, HEIGHT);

    // Salva come PPM
    FILE* out = fopen("frame_rgb.ppm", "wb");
    fprintf(out, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(rgb_buffer, 1, WIDTH * HEIGHT * 3, out);
    fclose(out);

    printf("✅ Immagine RGB salvata come frame_rgb.ppm\n");

    // Cleanup
    free(rgb_buffer);
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    munmap(buffer, buf.length);
    close(fd);
    return 0;
}
