#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "capture_camera.h"
#include "image_manipulation.cuh"
#include "conversion_CUDA.cuh"
#include "conversion.h"
#include "globals.h"




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

  gaussian_blur3<<<grid, block>>>(device_gray, device_blur, width, height);

  cudaMemcpy(device_prev_rgb, (unsigned char *)prev_rgb, width * height * 4, cudaMemcpyHostToDevice);
  gray_scale_conversion<<<num_blocks, num_threads>>>(device_prev_rgb, device_prev_gray, width, height);

  gaussian_blur3<<<grid, block>>>(device_prev_gray, device_prev_blur, width, height);

  compute_derivatives<<<grid, block>>>(device_prev_blur, device_blur ,d_Ix, d_Iy, d_It, width, height);

  size_t fsize = width * height * sizeof(float);

  int num_iterations = 100;
  float alpha = 10.0f;

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
        free(h_mag);

        if (best_magnitude < 20.0f) {
            best_x = prev_x;
            best_y = prev_y;
        }

        int box_size = 16 + fminf(best_magnitude * 2, 200.0f); 

        float square_alpha = 0.2f;

        float filtered_x = square_alpha * best_x + (1.0f - square_alpha) * prev_x;
        float filtered_y = square_alpha * best_y + (1.0f - square_alpha) * prev_y;

        prev_x = filtered_x;
        prev_y = filtered_y;

        draw_square((unsigned char*)rgb, width, height, (int)filtered_x, (int)filtered_y, box_size);
        
}

void scale_image_cn(unsigned char* rgb, int base_width, int base_height,unsigned char* scaled_image ,int window_width, int window_height) {

    cudaMemcpy(device_rgb, rgb, base_width * base_height * 4, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((window_width + 15) / 16, (window_height + 15) / 16);

    scale_image_cn_kernel<<<gridSize, blockSize>>>(device_rgb, base_width, base_height ,device_scaled_rgb, window_width, window_height);

    cudaMemcpy(scaled_image, device_scaled_rgb, window_height * window_width * 4, cudaMemcpyDeviceToHost);

}

void scale_image_bilinear(unsigned char* rgb, int base_width, int base_height,unsigned char* scaled_image ,int window_width, int window_height) {
    cudaMemcpy(device_rgb, rgb, base_width * base_height * 4, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((window_width + 15) / 16, (window_height + 15) / 16);

    scale_image_bilinear_kernel<<<gridSize, blockSize>>>(device_rgb, device_scaled_rgb, base_width, base_height, window_width, window_height);

    image_sharpen<<<gridSize, blockSize>>>(device_scaled_rgb, device_sharpened_rgb, window_width, window_height);

    cudaMemcpy(scaled_image, device_sharpened_rgb, window_height * window_width * 4, cudaMemcpyDeviceToHost);
}


void alloc_conversion(camera_t* camera) {
    cudaMalloc(&device_yuyv, camera->width * camera->height * 2);
    cudaMalloc(&device_rgb, camera->width * camera->height * 4);
}

void free_conversion(void) {
    cudaFree(device_rgb);
    cudaFree(device_yuyv);
}

void alloc_Edge(camera_t* camera) {
    if (!GPU) {
      cudaMalloc(&device_rgb, camera->width * camera->height * 4);
    }
    cudaMalloc(&device_gray, camera->width * camera->height);
    cudaMalloc(&device_output, camera->width * camera->height * 4);
}

void free_Edge(void) {
    cudaFree(device_rgb);
    cudaFree(device_gray);
    cudaFree(device_output);
}

void alloc_Optical(camera_t* camera) {
    size_t derivative_size = camera->width * camera->height * sizeof(float);
    size_t fsize = camera->width * camera->height * sizeof(float);

    if (!GPU) {
      cudaMalloc(&device_rgb, camera->width * camera->height * 4);
    }
    if (!EDGE_DET) {
      cudaMalloc(&device_gray, camera->width * camera->height);
    }

    cudaMalloc(&device_prev_gray, camera->width * camera->height);
    cudaMalloc(&device_prev_rgb, camera->width * camera->height *4);
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
    cudaMalloc(&device_blur, camera->width * camera->height);
    cudaMalloc(&device_prev_blur, camera->width * camera->height);
}


void free_Optical(void) {
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
    cudaFree(device_blur);
    cudaFree(device_prev_blur);
}