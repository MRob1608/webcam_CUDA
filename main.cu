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
#include <X11/Xatom.h>

extern Display* display;
extern Window window;
extern GC gc;
XEvent event;

int GPU = 1;
int EDGE_DET = 0;
int OPTICAL = 1;

unsigned char* device_rgb, *device_gray, *device_output, *device_yuyv, *device_blur;

unsigned char* device_prev_rgb, *device_prev_gray, *device_prev_blur;

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

  dim3 block(16, 16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);

  int num_pixels = width * height;
  int num_threads = 256;
  int num_blocks = (num_pixels + num_threads - 1) / num_threads;

  cudaMemcpy(device_rgb, (unsigned char *)rgb, width * height * 4, cudaMemcpyHostToDevice);
  gray_scale_conversion<<<num_blocks, num_threads>>>(device_rgb, device_gray, width, height);

  //blur_image<<<grid, block>>>(device_gray, device_blur, width, height);

  cudaMemcpy(device_prev_rgb, (unsigned char *)prev_rgb, width * height * 4, cudaMemcpyHostToDevice);
  gray_scale_conversion<<<num_blocks, num_threads>>>(device_prev_rgb, device_prev_gray, width, height);

  //blur_image<<<grid, block>>>(device_prev_gray, device_prev_blur, width, height);

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

        float* h_u = (float*)malloc(fsize);
        float* h_v = (float*)malloc(fsize);
        cudaMemcpy(h_u, d_u, fsize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v, d_v, fsize, cudaMemcpyDeviceToHost);

        float best_magnitude = 0.0f;
        int best_x = 0, best_y = 0;
        int block_size = 32;

        for (int by = 0; by < height; by += block_size) {
            for (int bx = 0; bx < width; bx += block_size) {
                float sum_u = 0.0f;
                float sum_v = 0.0f;
                int count = 0;

                for (int y = 0; y < block_size; y++) {
                    for (int x = 0; x < block_size; x++) {
                        int px = bx + x;
                        int py = by + y;
                        if (px >= width || py >= height) continue;
                        int idx = py * width + px;
                        sum_u += h_u[idx];
                        sum_v += h_v[idx];
                        count++;
                    }
                }

                if (count > 0) {
                    float avg_u = sum_u / count;
                    float avg_v = sum_v / count;
                    float magnitude = sqrtf(avg_u * avg_u + avg_v * avg_v);
                    if (magnitude > best_magnitude) {
                        best_magnitude = magnitude;
                        best_x = bx + block_size / 2;
                        best_y = by + block_size / 2;
                    }
                }
            }
        }

        free(h_u);
        free(h_v);

        int box_size = 16 + best_magnitude * 2; // minimo 16, massimo 48 se avg ≈ 1.0

        draw_square((unsigned char*)rgb, width, height, best_x, best_y, box_size);
        
}




int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("usage: <executable> <camera_device_name> - eg ./camera_capture /dev/video0\n");
    return -1;
  }

  printf("opening camera device [ %s ]\n", argv[1]);
  camera_t* camera = camera_open("/dev/video0", 640, 480);  //640 480

  init_x11(640,480);
  Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);

  camera_init(camera);
  camera_start(camera);


  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 100000;
  char image_name[1024];
  char* prev_rgb =  (char*)malloc(camera->width * camera->height * 4);

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

  cudaMalloc(&device_blur, camera->width * camera->height);
  cudaMalloc(&device_prev_blur, camera->width * camera->height);

  int i = 0;

  printf("cartes"); fflush(stdout);

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

  clock_gettime(CLOCK_MONOTONIC, &end_time); // dopo il ciclo
  double elapsed_sec = end_time.tv_sec - start_time.tv_sec +
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

  printf("Captured %d frames in %.2f seconds (%.2f FPS)\n", i, elapsed_sec, i / elapsed_sec);
  printf("done!\n");
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

  cudaFree(device_blur);
  cudaFree(device_prev_blur);
  


  printf("closing\n");
  camera_stop(camera);
  camera_finish(camera);
  camera_close(camera);
  return 0;
}

