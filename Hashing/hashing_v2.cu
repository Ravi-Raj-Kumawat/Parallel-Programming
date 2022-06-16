/*
    CUDA IMPLEMENTATION OF HASHTABLE:
    --> Kernels for simultaneous Insertion and Deletion of data are defined.
    --> For handling colisions, Open Addressing is used (Linear Probing).
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdint.h>
#include "Includes/gputimer.h"


#define TABLE_SIZE 1000
#define TABLE_BYTES TABLE_SIZE*sizeof(uint32_t)

#define INPUT_ARRAY_SIZE 1000
#define INPUT_ARRAY_BYTES INPUT_ARRAY_SIZE*sizeof(uint32_t)
#define INPUT_THREDS_NUM 1000
#define INPUT_BLOCKS_NUM (INPUT_ARRAY_SIZE/INPUT_THREDS_NUM)

#define DELETE_ARRAY_SIZE 100
#define DELETE_ARRAY_BYTES DELETE_ARRAY_SIZE*sizeof(uint32_t)
#define DELETE_THREDS_NUM 100
#define DELETE_BLOCKS_NUM (DELETE_ARRAY_SIZE/DELETE_THREDS_NUM)

#define NULL_DATA 0
#define DELETED_DATA INT_MAX

__device__ __host__ uint32_t hash(uint32_t n)
{
    /*
    A simple hash function to be used in our HashTable
    */
    return (n * n) % TABLE_SIZE;
}

__device__ __host__ void initializeHashTable(uint32_t *hm)
{
    for (uint32_t i = 0; i < TABLE_SIZE; i++)
    {
        hm[i] = NULL_DATA;
    }
}

void printHashTable(uint32_t *hm)
{
    /*
    This function will output the contents of a HashTable in the form:   KEY   -->   VALUE
    */
    printf("Printing the content of HashTable:-\n");
    printf("\tKEY\t-->\tVALUE\n");
    for (uint32_t i = 0; i < TABLE_SIZE; i++)
    {
        printf("\t%li\t-->\t", i);
        if(hm[i] == DELETED_DATA){
            printf("<Deleted>\n");
            continue;
        }
        if (hm[i] != NULL_DATA)
            printf("%lu\n", hm[i]);
        else
            printf("-\n");
    }
}

__global__ void parallel_insert(uint32_t *d_input, uint32_t *d_hm)
{
    /*
    To insert an array of data to HashTable all at once using GPU
    */
    uint32_t input_idx = blockIdx.x * INPUT_THREDS_NUM + threadIdx.x;
    uint32_t table_idx = hash(d_input[input_idx]);
    // __syncthreads();
    while (table_idx < TABLE_SIZE){
        if(d_hm[table_idx] == NULL_DATA || d_hm[table_idx] == DELETED_DATA){
            d_hm[table_idx] = d_input[input_idx];
            return;
        }
        table_idx++;
    }
    if(table_idx == TABLE_SIZE){
        printf("No more Space! (Data = %li)\n", d_input[input_idx]);
        return;
    }
}

__global__ void parallel_delete(uint32_t *d_del, uint32_t* d_hm){
    uint32_t del_idx = blockIdx.x*DELETE_THREDS_NUM + threadIdx.x;
    uint32_t hm_idx = hash(d_del[del_idx]);
    while(hm_idx < TABLE_SIZE && d_hm[hm_idx] != NULL_DATA){
        if(d_hm[hm_idx] == d_del[del_idx]){
            d_hm[hm_idx] = DELETED_DATA;
            return;
        }
    }
    printf("Not Found!\n");
}

int main(uint32_t argc, char **argv)
{
    GpuTimer timer;

    // Variables in CPU
    printf("Initialising Hash-Table....\n");
    uint32_t *h_input = (uint32_t *)malloc(INPUT_ARRAY_BYTES); // Input Array
    uint32_t *h_delete = (uint32_t *)malloc(DELETE_ARRAY_BYTES); // Input Array
    uint32_t *h_hm = (uint32_t *)malloc(TABLE_BYTES); // Hashtable in Host Memory
    initializeHashTable(h_hm);

    // Variables in GPU
    uint32_t *d_input;
    uint32_t *d_delete;
    uint32_t *d_hm; // Hash Table in Device/GPU Memory
    cudaMalloc((void **)&d_input, INPUT_ARRAY_BYTES);
    cudaMalloc((void **)&d_delete, DELETE_ARRAY_BYTES);
    cudaMalloc((void **)&d_hm, TABLE_BYTES);

    printf("\n");
    printf("Inserting an array sorted in ASCENDING ORDER...\n");
    for(uint32_t i = 0; i < INPUT_ARRAY_SIZE; i++){
        h_input[i] = i+1;
    }
    cudaMemcpy(d_input, h_input, INPUT_ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hm, h_hm, INPUT_ARRAY_BYTES, cudaMemcpyHostToDevice);
    timer.Start();
    parallel_insert<<<INPUT_BLOCKS_NUM, INPUT_THREDS_NUM>>>(d_input, d_hm);
    timer.Stop();
    cudaMemcpy(h_hm, d_hm, TABLE_BYTES, cudaMemcpyDeviceToHost);
    printf("Insertion Completed!\n");
    printf("Time taken = %g ms\n", timer.Elapsed());

    printf("\n");
    // Printing the resulting HashTable
    printf("The Resulting Hash Table is saved in 'output.txt' file.\n");
    freopen("output.txt", "w", stdout);
    printHashTable(h_hm);

    free(h_input);
    free(h_delete);
    free(h_hm);
    cudaFree(d_input);
    cudaFree(d_delete);
    cudaFree(d_hm);
    return 0;
}