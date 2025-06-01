extern "C" {
#include "interface.h"
#include "conversion.h"
}
#include "capture_camera.h"
#include "conversion_CUDA.cuh"
#include "image_manipulation.cuh"
#include "wrappers.cuh"
#include "globals.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <asm/types.h>
#include <linux/videodev2.h>

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <time.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>

extern Display* display;
extern Window window;
extern GC gc;
XEvent event;


int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("usage: <executable> <camera_device_name> - eg ./camera_capture /dev/video0\n");
    return -1;
  }

  printf("opening camera device [ %s ]\n", argv[1]);
  camera_t* camera = camera_open((const char*)argv[1]);  

  int fps = select_camera_format(camera);

  if (fps == -1) return -1;

  camera_init(camera, fps);
  camera_start(camera);

  init_x11(camera->width,camera->height);
  Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);


  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 100000;
  char image_name[1024];


  char* prev_rgb =  (char*)malloc(camera->width * camera->height * 4);

  if(GPU) alloc_conversion(camera);
  
  if (EDGE_DET) alloc_Edge(camera);

  if (OPTICAL) alloc_Optical(camera);
  
  int i = 0;

  while(1) {
    while (XPending(display)) {
        XNextEvent(display, &event);
        if (event.type == ClientMessage) {
            if ((Atom)event.xclient.data.l[0] == wm_delete_window) {
                printf("\nFinestra chiusa\n");
                goto exit_loop;
            }
        }
    }


    if (camera_frame(camera, timeout)>0) {
      sprintf(image_name, "image-%05d.pgm", i);
      printf("\racquiring frame [ %05d ]", i);
      fflush(stdout);
      char* rgb =  (char*)malloc(camera->width * camera->height * 4);
      
      convert_yuyv_to_bgra(camera,rgb, GPU);


      if (EDGE_DET) {
        apply_edge_detection(rgb, camera->width, camera->height);
      }

      if (OPTICAL && prev_rgb != NULL) {
        apply_optical_flow(rgb, prev_rgb, camera->width, camera->height);
      }

      
      mirror_image((unsigned char*)rgb, camera->height, camera->width);
      //savePGM(camera, image_name);
      display_frame((unsigned char*)rgb,camera->width, camera->height);
      memcpy(prev_rgb, rgb, camera->width * camera->height * 4);
      free(rgb);
    }
    i++;
  }
  exit_loop:
  XDestroyWindow(display, window);
  XCloseDisplay(display);

  clock_gettime(CLOCK_MONOTONIC, &end_time); 
  double elapsed_sec = end_time.tv_sec - start_time.tv_sec +
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

  printf("Captured %d frames in %.2f seconds (%.2f FPS)\n", i, elapsed_sec, i / elapsed_sec);
  printf("done!\n");
  camera_frame(camera, timeout);

  if (GPU) free_conversion();
  
  if(EDGE_DET) free_conversion();

  if (OPTICAL) free_Optical();

  free(prev_rgb);
  printf("closing\n");
  camera_stop(camera);
  camera_finish(camera);
  camera_close(camera);
  return 0;
}

