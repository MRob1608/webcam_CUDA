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
int EDGE_DET = 0;
int OPTICAL = 1;

unsigned char* device_rgb, *device_gray, *device_output, *device_yuyv;

unsigned char* device_prev_rgb, *device_prev_gray;

float* d_Ix, *d_It, *d_Iy, *d_u, *d_v, *d_u_avg, *d_v_avg, *mag;


void draw_square(
    unsigned char* image,
    int width, int height,
    int center_x, int center_y,
    int size
) {
    int half = size / 2;
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            int px = center_x + x;
            int py = center_y + y;
            if (px >= 0 && px < width && py >= 0 && py < height && ((x == -half or x ==half) || (y == -half or y ==half))) {
                int idx = (py * width + px) * 4; // BGRA
                image[idx + 0] = 0;   // B
                image[idx + 1] = 255; // G
                image[idx + 2] = 0;   // R
                image[idx + 3] = 0; // A
            }
        }
    }
}

void convert_yuyv_to_bgra(camera_t* camera, char* rgb, int use_gpu) { 
  if (use_gpu) {


        int num_pixels = camera->width * camera->height;
        int num_threads = 256;
        int num_blocks = (num_pixels / 2 + num_threads - 1) / num_threads;

        cudaMemcpy(device_yuyv, camera->head.start, camera->width * camera->height * 2, cudaMemcpyHostToDevice);

        yuyv_to_bgr_CUDA<<<num_blocks, num_threads>>>(device_yuyv, device_rgb, camera->height, camera->width);

        cudaMemcpy(rgb, device_rgb, camera->width * camera->height * 4, cudaMemcpyDeviceToHost);

      } else {
        yuyv_to_bgr(camera->head.start,(unsigned char *)rgb,camera->width, camera->height);
      }
}

void apply_edge_detection(char* rgb, int width, int height) {
  cudaMemcpy(device_rgb, (unsigned char *)rgb, width * height * 4, cudaMemcpyHostToDevice);

  int num_pixels = width * height;
  int num_threads = 256;
  int num_blocks = (num_pixels + num_threads - 1) / num_threads;
        
  gray_scale_conversion<<<num_blocks, num_threads>>>(device_rgb, device_gray, width, height);

  dim3 blockSize(16, 16);  // 256 thread per blocco
  dim3 gridSize(
      (width + blockSize.x - 1) / blockSize.x,   // ceil(width / 16)
      (height + blockSize.y - 1) / blockSize.y   // ceil(height / 16)
  );


  edge_detection_overlay<<<gridSize, blockSize>>>(device_rgb, device_output, device_gray, width, height);

  cudaMemcpy((unsigned char*)rgb, device_output, width * height * 4, cudaMemcpyDeviceToHost);

}


void apply_optical_flow(char* rgb, char* prev_rgb, int width, int height) {

  int num_pixels = width * height;
  int num_threads = 256;
  int num_blocks = (num_pixels + num_threads - 1) / num_threads;

  cudaMemcpy(device_rgb, (unsigned char *)rgb, width * height * 4, cudaMemcpyHostToDevice);
  gray_scale_conversion<<<num_blocks, num_threads>>>(device_rgb, device_gray, width, height);

  cudaMemcpy(device_prev_rgb, (unsigned char *)prev_rgb, width * height * 4, cudaMemcpyHostToDevice);
  gray_scale_conversion<<<num_blocks, num_threads>>>(device_prev_rgb, device_prev_gray, width, height);

  dim3 block(16, 16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);

  compute_derivatives<<<grid, block>>>(device_prev_gray, device_gray ,d_Ix, d_Iy, d_It,width, height);

  size_t fsize = width * height * sizeof(float);

  int num_iterations = 100;
  float alpha = 15.0f;

  for (int i = 0; i < num_iterations; i++) {
      // 1. Calcola media locale
      average_uv<<<grid, block>>>(
          d_u, d_v,
          d_u_avg, d_v_avg,
          width, height
      );

            // 2. Aggiorna u, v
            update_uv<<<grid, block>>>(
                d_Ix, d_Iy, d_It,
                d_u_avg, d_v_avg,
                d_u, d_v,
                alpha,
                width, height
            );
        }

        compute_flow_magnitude<<<grid, block>>>(d_u, d_v, mag, width, height);

        float* h_mag = (float*)malloc(fsize);
        cudaMemcpy(h_mag, mag, fsize, cudaMemcpyDeviceToHost);

        int max_idx = 0;
        float max_val = 0.0f;

        for (int i = 0; i < width * height; i++) {
            if (h_mag[i] > max_val) {
                max_val = h_mag[i];
                max_idx = i;
            }
        }

        int max_x = max_idx % width;
        int max_y = max_idx / width;

        draw_square((unsigned char*)rgb, width, height, max_x, max_y, 50);
        
}




int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("usage: <executable> <camera_device_name> <number_of_frames> - eg ./camera_capture /dev/video0 100\n");
    return -1;
  }

  printf("opening camera device [ %s ]\n", argv[1]);
  camera_t* camera = camera_open("/dev/video0", 640, 480);  //640 480

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
  char* prev_rgb =  (char*)malloc(camera->width * camera->height * 4);

  //unsigned char *device_yuyv, *device_rgb;

  if(GPU) {
    cudaMalloc(&device_yuyv, camera->width * camera->height * 2);
    cudaMalloc(&device_rgb, camera->width * camera->height * 4);
  }

  if (EDGE_DET) {
    if (!GPU) {
      cudaMalloc(&device_rgb, camera->width * camera->height * 4);
    }
    cudaMalloc(&device_gray, camera->width * camera->height);
    cudaMalloc(&device_output, camera->width * camera->height * 4);
  }

  if (OPTICAL) {
    size_t derivative_size = camera->width * camera->height * sizeof(float);
    size_t fsize = camera->width * camera->height * sizeof(float);

    if (!GPU) {
      cudaMalloc(&device_rgb, camera->width * camera->height * 4);
    }
    if (!EDGE_DET) {
      cudaMalloc(&device_gray, camera->width * camera->height);
    }

    cudaMalloc(&device_prev_gray, camera->width * camera->height);
    cudaMalloc(&device_prev_rgb, camera->width * camera->height);
    cudaMalloc(&d_Ix, derivative_size);
    cudaMalloc(&d_Iy, derivative_size);
    cudaMalloc(&d_It, derivative_size);
    cudaMalloc(&d_u, fsize);
    cudaMalloc(&d_v, fsize);
    cudaMemset(d_u, 0, fsize);  
    cudaMemset(d_v, 0, fsize);
    cudaMalloc(&d_u_avg, fsize);
    cudaMalloc(&d_v_avg, fsize);
    cudaMalloc(&mag,camera->width * camera->height * sizeof(float));
  }

  for (int i = 0; i < num_frames; ++i) {
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
  }
  clock_gettime(CLOCK_MONOTONIC, &end_time); // dopo il ciclo
  double elapsed_sec = end_time.tv_sec - start_time.tv_sec +
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

  printf("\nCaptured %d frames in %.2f seconds (%.2f FPS)\n", num_frames, elapsed_sec, num_frames / elapsed_sec);
  printf("\ndone!\n");
  camera_frame(camera, timeout);

  if (GPU) {
    cudaFree(device_rgb);
    cudaFree(device_yuyv);
  }
  
  if(EDGE_DET) {
    cudaFree(device_rgb);
    cudaFree(device_gray);
    cudaFree(device_output);
  }

  if (OPTICAL) {
    cudaFree(device_rgb);
    cudaFree(device_prev_rgb);
    cudaFree(device_gray);
    cudaFree(device_prev_gray);
    cudaFree(d_It);
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_u_avg);
    cudaFree(d_v_avg);
    cudaFree(mag);
  } 


  printf("closing\n");
  camera_stop(camera);
  camera_finish(camera);
  camera_close(camera);
  return 0;
}