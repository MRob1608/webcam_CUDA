#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_cuda() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_cuda<<<1, 1>>>();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();  // forza completamento kernel
    fflush(stdout);           // forza stampa terminale
    return 0;
}
