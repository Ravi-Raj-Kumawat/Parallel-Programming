#include <stdio.h>
#include <cuda.h>
#include "Includes/sha256.h"

__global__ void hash(float* d_out, float* d_in){  // Declaring a Kernal
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f*f*f;
}

int main(int argc, char** argv){
    const int ARRAY_SIZE = 96;
    const int ARRAY_BYTES = ARRAY_SIZE*sizeof(float);

    float h_in[ARRAY_SIZE]; // h = HOST = CPU
    float h_out[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)  // INPUT ARRAY
    {
        h_in[i] = float(i);
    }
    
    // Declaring GPU memory pointers
    float * d_in; // d = DEVICE = GPU
    float * d_out;

    // Allocate GPU Memory
    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);

    // Transfer the array CPU -> GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    //       destinaton source

    // Lanch the Kernel
    hash<<<1, ARRAY_SIZE>>>(d_out, d_in);
    //      Lanch Parameters    Arguments

    // Transfer the result GPU -> CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Printing Result
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        printf("%f ", h_out[i]);
        printf((i %4 == 3)? "\n": "\t");
    }
    
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}