#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Calulates the index of a pixel in BGRA starting from the coordinates x and y
__device__ int idx(int x, int y, int width) {
    return 4 * (y * width + x);
}

//Applies a simple 3x3 blur to the image in grayscale
__global__ void blur_image(unsigned char* gray, unsigned char* blur, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)  //escludo i pixel ai bordi
        return;

    int pixel_idx = y * width + x;
    int sum = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int p_id = (y+j)*width + x + i;
            sum += gray[p_id];        
        }
    }
    sum /= 9;
    blur[pixel_idx] = sum;
}

//Applies Gaussian blur to the image in grayscale
__global__ void gaussian_blur3(unsigned char* gray, unsigned char* blur, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 0 || y < 0 || x >= width  || y >= height )  
        return;

    int pixel_idx = y * width + x;

    const int blr[3][3] = {  //Gaussian blur matrix
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    int sum = 0;
    int den = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int xi = x + j;
            int yi = y + i;

            if (xi >= 0 && xi < width && yi >= 0 && yi < height) {
                int weight = blr[i + 1][j + 1];
                sum += weight * gray[yi * width + xi];
                den += weight;
            }
        }
    }

    
    blur[pixel_idx] = sum / den;

}

//Convert a BGRA image into a grayscale image (values from 0 to 255)
__global__ void gray_scale_conversion(unsigned char* rgb, unsigned char* gray, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= height * width) return;

    int pixel_idx = 4*tid;

    gray[tid] = rgb[pixel_idx] * 0.114f + rgb[pixel_idx + 1] * 0.587f + rgb[pixel_idx + 2] * 0.299f;

}

//Applies edge detection to the image using two matrix for vertical and horizontal edges
__global__ void edge_detection_overlay(unsigned char* rgb, unsigned char* output, unsigned char* grayscale, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)  //escludo i pixel ai bordi
        return;

    float Gx = 0.0f, Gy = 0.0f;
    int thold = 50;

    const int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    const int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int ni = x + i;
            int nj = y + j;
            float gray = grayscale[nj * width + ni];

            Gx += gray * sobel_x[j+1][i+1];
            Gy += gray * sobel_y[j+1][i+1];
        }
    }

    float magnitude = sqrtf(Gx*Gx + Gy*Gy);
    if (magnitude > 255) magnitude = 255;

    int pixel_idx = idx(x,y,width);

    if (magnitude > thold) {
        output[pixel_idx] = rgb[pixel_idx];
        output[pixel_idx + 1] = rgb[pixel_idx+1];
        output[pixel_idx + 2] = 255;  //Setting the color of the pixel to red if is detected as an edge
        output[pixel_idx + 3] = 0;
    } else {
        output[pixel_idx]     = rgb[pixel_idx];
        output[pixel_idx + 1] = rgb[pixel_idx+1];
        output[pixel_idx + 2] = rgb[pixel_idx+2];
        output[pixel_idx + 3] = 0;
    }
}

//Create a green square given its center coordinates and size
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

//Calculates three values for each pixel consisting in the x_derivative, y_derivetive and time derivative
__global__ void compute_derivatives(
    const unsigned char* prev_gray,
    const unsigned char* curr_gray,
    float* Ix, float* Iy, float* It,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    int idx = y * width + x;

    int idx_left   = y * width + (x - 1);
    int idx_right  = y * width + (x + 1);
    int idx_top    = (y - 1) * width + x;
    int idx_bottom = (y + 1) * width + x;

    // Averages between curr and prev
    //float avg_center  = 0.5f * (curr_gray[idx] + prev_gray[idx]);
    float avg_left    = 0.5f * (curr_gray[idx_left] + prev_gray[idx_left]);
    float avg_right   = 0.5f * (curr_gray[idx_right] + prev_gray[idx_right]);
    float avg_top     = 0.5f * (curr_gray[idx_top] + prev_gray[idx_top]);
    float avg_bottom  = 0.5f * (curr_gray[idx_bottom] + prev_gray[idx_bottom]);

    // Central spacial derivatives
    Ix[idx] = (avg_right - avg_left) * 0.5f;
    Iy[idx] = (avg_bottom - avg_top) * 0.5f;

    // Time derivative
    It[idx] = (float)(curr_gray[idx]) - (float)(prev_gray[idx]);
}

//Kernel for smoothing the flow, calculating the u_avg,v_avg for a pixel starting from its neighbors
__global__ void average_uv(
    const float* u_in, const float* v_in,
    float* u_avg, float* v_avg,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    int idx      = y * width + x;
    int idx_left  = y * width + (x - 1);
    int idx_right = y * width + (x + 1);
    int idx_top   = (y - 1) * width + x;
    int idx_bottom= (y + 1) * width + x;

    u_avg[idx] = 0.25f * (u_in[idx_left] + u_in[idx_right] + u_in[idx_top] + u_in[idx_bottom]);
    v_avg[idx] = 0.25f * (v_in[idx_left] + v_in[idx_right] + v_in[idx_top] + v_in[idx_bottom]);
}

//Updates the u, v values using the derivatives Ix, Iy, Iz and the averages previosly calculated
__global__ void update_uv(
    const float* Ix, const float* Iy, const float* It,
    const float* u_avg, const float* v_avg,
    float* u, float* v,
    float alpha,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    int idx = y * width + x;

    float ix = Ix[idx];
    float iy = Iy[idx];
    float it = It[idx];

    float ubar = u_avg[idx];
    float vbar = v_avg[idx];

    float denom = alpha * alpha + ix * ix + iy * iy;

    if (denom != 0.0f) {
        float term = (ix * ubar + iy * vbar + it) / denom;
        u[idx] = ubar - ix * term;
        v[idx] = vbar - iy * term;
    }
}

//Calculates the magnitude of the flow for every pixel
__global__ void compute_flow_magnitude(const float* u, const float* v, float* mag, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thold = 2;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float dx = u[idx];
    float dy = v[idx];
    float tmp_mag = sqrtf(dx * dx + dy * dy);
    if (tmp_mag < thold) tmp_mag = 0;
    mag[idx] = tmp_mag;
}

//Scales the image dinamically using the closest neighbor technique
__global__ void scale_image_cn_kernel(
    const unsigned char* input,
    int src_width, int src_height,
    unsigned char* output,
    int dst_width, int dst_height
) {
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_out >= dst_width || y_out >= dst_height) return;

    int x_in = x_out * src_width / dst_width;
    int y_in = y_out * src_height / dst_height;

    int src_idx = (y_in * src_width + x_in) * 4;
    int dst_idx = (y_out * dst_width + x_out) * 4;

    output[dst_idx + 0] = input[src_idx + 0];  // B
    output[dst_idx + 1] = input[src_idx + 1];  // G
    output[dst_idx + 2] = input[src_idx + 2];  // R
    output[dst_idx + 3] = input[src_idx + 3];  // A
}

// Scales the image dinamically using the bilinear tecnique
__global__ void scale_image_bilinear_kernel(
    unsigned char* input, unsigned char* output,
    int in_width, int in_height,
    int out_width, int out_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height) return;

    float scale_x = (float)in_width / out_width;
    float scale_y = (float)in_height / out_height;

    float src_x = x * scale_x;
    float src_y = y * scale_y;

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = min(x0 + 1, in_width - 1);
    int y1 = min(y0 + 1, in_height - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;

    
    int idx_A = idx(x0,y0, in_width);
    int idx_B = idx(x1,y0, in_width);
    int idx_C = idx(x0,y1, in_width);
    int idx_D = idx(x1,y1, in_width);

    
    int out_idx = idx(x,y,out_width);

    for (int c = 0; c < 4; c++) {  
        float A = input[idx_A + c];
        float B = input[idx_B + c];
        float C = input[idx_C + c];
        float D = input[idx_D + c];

        float top = A * (1.0f - dx) + B * dx;
        float bottom = C * (1.0f - dx) + D * dx;
        float value = top * (1.0f - dy) + bottom * dy;

        output[out_idx + c] = (unsigned char)(value + 0.5f);
    }
}

//Applies a sharpening filter to the image (used to prevent unwanted blur using bilinear scaling)
__global__ void image_sharpen(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 0 || y < 0 || x >= width  || y >= height )  
        return;

    int pixel_idx = idx(x,y,width);

    const int shp[3][3] = {
        {0, -1, 0},
        {-1, 5,-1},
        {0, -1, 0}
    };

    for (int c = 0; c < 3; c++) {
        float sum = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int dx = fmaxf(0, fminf(x+i, width-1));
                int dy = fmaxf(0, fminf(y+j, height-1));
                int id = idx(dx, dy, width);
                sum += shp[j+1][i+1] * input[id+c];
            }
        }
        output[pixel_idx+c] = (unsigned char)fminf(fmaxf(sum+0.5f, 0.0f), 255.0f);
    }

    output[pixel_idx+3] = input[pixel_idx+3];
}



