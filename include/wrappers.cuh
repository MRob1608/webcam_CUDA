#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "capture_camera.h"



void convert_yuyv_to_bgra(camera_t* camera, char* rgb, int use_gpu);

void apply_edge_detection(char* rgb, int width, int height);

void apply_optical_flow(char* rgb, char* prev_rgb, int width, int height);

void scale_image_cn(unsigned char* rgb, int base_width, int base_height,unsigned char* scaled_image ,int window_width, int window_height);

void scale_image_bilinear(unsigned char* rgb, int base_width, int base_height,unsigned char* scaled_image ,int window_width, int window_height);

void alloc_conversion(camera_t* camera);

void free_conversion(void);

void alloc_Edge(camera_t* camera);

void free_Edge(void);

void alloc_Optical(camera_t* camera);

void free_Optical(void);