extern "C" {
#include "capture_camera.h"
#include "interface.h"
#include "conversion.h"
}
#include "conversion_CUDA.cuh"
#include "image_manipulation.cuh"

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

extern Display* display;
extern Window window;
extern GC gc;

int GPU = 1;
int EDGE_DET = 1;


int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("usage: <executable> <camera_device_name> <number_of_frames> - eg ./camera_capture /dev/video0 100\n");
    return -1;
  }

  printf("opening camera device [ %s ]\n", argv[1]);
  camera_t* camera = camera_open("/dev/video0", 640, 480);

  init_x11(640,480);
  camera_init(camera);
  camera_start(camera);

  const int num_frames = atoi(argv[2]);
  if (num_frames < 0) {
    printf("error, invalid number of frames - it must be positive :)\n");
    printf("usage: <executable> <camera_device_name> <number_of_frames> - eg ./camera_capture /dev/video0 100\n");
    return -1;
  }

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  printf("capturing [ %05d ] frames\n", num_frames);
  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 100000;
  char image_name[1024];
  
  if (GPU) {
    printf("Conversione YUYV in RGB con GPU\n"); fflush(stdout);
  } else {
    printf("Conversione YUYV in RGB con CPU\n"); fflush(stdout);
  }


  for (int i = 0; i < num_frames; ++i) {
    if (camera_frame(camera, timeout)>0) {
      sprintf(image_name, "image-%05d.pgm", i);
      printf("\racquiring frame [ %05d ]", i);
      fflush(stdout);
      char* rgb =  (char*)malloc(camera->width * camera->height * 4);

      if (GPU) {

        unsigned char *device_yuyv, *device_rgb;

        int num_pixels = camera->width * camera->height;
        int num_threads = 256;
        int num_blocks = (num_pixels / 2 + num_threads - 1) / num_threads;


        cudaMalloc(&device_yuyv, camera->width * camera->height * 2);
        cudaMalloc(&device_rgb, camera->width * camera->height * 4);
        cudaMemcpy(device_yuyv, camera->head.start, camera->width * camera->height * 2, cudaMemcpyHostToDevice);

        yuyv_to_bgr_CUDA<<<num_blocks, num_threads>>>(device_yuyv, device_rgb, camera->height, camera->width);

        cudaMemcpy((unsigned char*)rgb, device_rgb, camera->width * camera->height * 4, cudaMemcpyDeviceToHost);
        cudaFree(device_rgb);
        cudaFree(device_yuyv);

      } else {
        yuyv_to_bgr(camera->head.start,(unsigned char*)rgb,camera->width, camera->height);
      }


      if (EDGE_DET) {
        unsigned char* device_rgb, *device_gray, *device_output;

        cudaMalloc(&device_rgb, camera->width * camera->height * 4);
        cudaMalloc(&device_gray, camera->width * camera->height);
        cudaMemcpy(device_rgb, (unsigned char *)rgb, camera->width * camera->height * 4, cudaMemcpyHostToDevice);

        int num_pixels = camera->width * camera->height;
        int num_threads = 256;
        int num_blocks = (num_pixels + num_threads - 1) / num_threads;
        
        gray_scale_conversion<<<num_blocks, num_threads>>>(device_rgb, device_gray, camera->width, camera->height);

        dim3 blockSize(16, 16);  // 256 thread per blocco
        dim3 gridSize(
            (camera->width + blockSize.x - 1) / blockSize.x,   // ceil(width / 16)
            (camera->height + blockSize.y - 1) / blockSize.y   // ceil(height / 16)
        );

        cudaMalloc(&device_output, camera->width * camera->height * 4);

        edge_detection_overlay<<<gridSize, blockSize>>>(device_rgb, device_output, device_gray, camera->width, camera->height);

        cudaMemcpy((unsigned char*)rgb, device_output, camera->width * camera->height * 4, cudaMemcpyDeviceToHost);

        cudaFree(device_rgb);
        cudaFree(device_gray);
        cudaFree(device_output);
      }

      
      mirror_image((unsigned char*)rgb, camera->height, camera->width);
      //savePGM(camera, image_name);
      display_frame((unsigned char*)rgb,camera->width, camera->height);
      
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end_time); // dopo il ciclo
  double elapsed_sec = end_time.tv_sec - start_time.tv_sec +
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

  printf("\nCaptured %d frames in %.2f seconds (%.2f FPS)\n", num_frames, elapsed_sec, num_frames / elapsed_sec);
  printf("\ndone!\n");
  camera_frame(camera, timeout);

  printf("closing\n");
  camera_stop(camera);
  camera_finish(camera);
  camera_close(camera);
  return 0;
}