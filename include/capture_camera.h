#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
  uint8_t* start;
  size_t length;
} buffer_t;

typedef struct {
  int fd;
  uint32_t width;
  uint32_t height;
  buffer_t head;        // buffer for the current image

  size_t buffer_count;
  buffer_t* buffers;    // image buffers four nimage buffers
} camera_t;

void quit(const char * msg);

int xioctl(int fd, int request, void* arg);

/*
  Opens the camera device and stores the requested image size in the camera struct
*/
camera_t* camera_open(const char * device, uint32_t width, uint32_t height);

/*
  1. queries the capability of he camera
  2. checks if device supports cropping
  3. allocates memory buffers for dma operation
  4. sets up mmap with the requested buffers
*/
void camera_init(camera_t* camera);

// starts the streaming
void camera_start(camera_t* camera);

// stops the streaming
void camera_stop(camera_t* camera);

// unmaps the buffers
void camera_finish(camera_t* camera);

// closes the device
void camera_close(camera_t* camera);

// captures a frame from the current buffer
int camera_capture(camera_t* camera);

int camera_frame(camera_t* camera, struct timeval timeout);

void savePGM(camera_t* camera, char* filename);