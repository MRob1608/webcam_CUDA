extern "C" {
#include "interface.h"
#include "conversion.h"
}
#include "capture_camera.h"
#include "conversion_CUDA.cuh"
#include "image_manipulation.cuh"
#include "wrappers.cuh"
#include "globals.h"
#include "rotation.cuh"

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

#define MAX_WIDTH  3840
#define MAX_HEIGHT 2160

extern Display* display;
extern Window window;
extern GC gc;
XEvent event;
XWindowAttributes attr;


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

  int rotated_width = camera->width;
  int rotated_height = camera->height;

  if (ROTATION_ANGLE == 90 || ROTATION_ANGLE == 270) {
    rotated_width = camera->height;
    rotated_height = camera->width;
  }

  init_x11(rotated_width,rotated_height);  //Creating the window
  Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);  //Setting response to closing


  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);  //Staring a timer

  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 100000;
  char image_name[1024];


  char* prev_rgb =  (char*)malloc(camera->width * camera->height * 4);  

  if(GPU) alloc_conversion(camera);  //Wrapper for the allocation of all the global variables needed in the procedures
  
  if (EDGE_DET) alloc_Edge(camera);

  if (OPTICAL) alloc_Optical(camera);

  if (ROTATION_ANGLE) alloc_Rotation(camera);

  cudaMalloc(&device_scaled_rgb, MAX_WIDTH * MAX_HEIGHT * 4);  
  cudaMalloc(&device_sharpened_rgb, MAX_WIDTH * MAX_HEIGHT * 4);
  
  int i = 0;

  while(1) {
    
    //Processing all the events on the window and exiting the loop in case of closed window
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
      
      convert_yuyv_to_bgra(camera,rgb, GPU);  //Converting the image to BGRA


      if (EDGE_DET) {
        apply_edge_detection(rgb, camera->width, camera->height);
      }

      if (OPTICAL && prev_rgb != NULL) {
        apply_optical_flow(rgb, prev_rgb, camera->width, camera->height);
      }

      if (GPU) {
        mirror_image_gpu((unsigned char*)rgb, camera->width, camera->height);
      } else {
        mirror_image((unsigned char*)rgb, camera->height, camera->width);
      }

      if (ROTATION_ANGLE) {
        rotate_image((unsigned char*)rgb, camera->width, camera->height, ROTATION_ANGLE);
      }

      //Getting the actual size of the window
      XGetWindowAttributes(display, window, &attr);
      int window_width = attr.width;
      int window_height = attr.height;

      if (window_width == rotated_width && window_height == rotated_height) {
        //Window never chenged size
        display_frame((unsigned char*)rgb,rotated_width, rotated_height);
        memcpy(prev_rgb, rgb, rotated_width * rotated_height * 4);
        free(rgb);
      } else {

        char* scaled_image = (char*)malloc(window_height * window_width * 4);  //Allocating a buffer for the scaled image

        if (BILINEAR) {
          scale_image_bilinear((unsigned char*)rgb, rotated_width, rotated_height,(unsigned char*) scaled_image, window_width, window_height);
        } else {
          scale_image_cn((unsigned char*)rgb, rotated_width, rotated_height,(unsigned char*) scaled_image, window_width, window_height);
        }

        display_frame((unsigned char*)scaled_image, window_width, window_height);
        memcpy(prev_rgb, rgb, camera->width * camera->height * 4);
        free(rgb);
        free(scaled_image);
      }

      
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

  if (GPU) free_conversion();  //Wrappers to free all the used variables
  
  if(EDGE_DET) free_conversion();

  if (OPTICAL) free_Optical();

  if (ROTATION_ANGLE) free_Rotation();

  cudaFree(device_scaled_rgb);
  cudaFree(device_sharpened_rgb);

  free(prev_rgb);
  printf("closing\n");
  camera_stop(camera);
  camera_finish(camera);
  camera_close(camera);
  return 0;
}

