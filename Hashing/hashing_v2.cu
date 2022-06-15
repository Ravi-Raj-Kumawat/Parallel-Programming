/*
    CUDA IMPLEMENTATION OF HASHTABLE:
    --> Kernels for simultaneous Insertion and Deletion of data are defined.
    --> For handling colisions, Open Addressing is used (Linear Probing).
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include "Includes/gputimer.h"


#define TABLE_SIZE 1000
#define TABLE_BYTES TABLE_SIZE*sizeof(int)

#define INPUT_ARRAY_SIZE 100
#define INPUT_ARRAY_BYTES INPUT_ARRAY_SIZE*sizeof(int)
#define INPUT_THREDS_NUM 100
#define INPUT_BLOCKS_NUM (INPUT_ARRAY_SIZE/INPUT_THREDS_NUM)

#define DELETE_ARRAY_SIZE 10
#define DELETE_ARRAY_BYTES DELETE_ARRAY_SIZE*sizeof(int)
#define DELETE_THREDS_NUM 100
#define DELETE_BLOCKS_NUM (DELETE_ARRAY_SIZE/DELETE_THREDS_NUM)

#define NULL_DATA NULL
#define DELETED_DATA LONG_MAX

__device__ __host__ long hash(long n)
{
    /*
    A simple hash function to be used in our HashTable
    */
    return (n * n) % TABLE_SIZE;
}

__device__ __host__ void initializeHashTable(long *hm)
{
    for (long i = 0; i < TABLE_SIZE; i++)
    {
        hm[i] = NULL_DATA;
    }
}

void printHashTable(long *hm)
{
    /*
    This function will output the contents of a HashTable in the form:   KEY   -->   VALUE
    */
    printf("Printing the content of HashTable:-\n");
    printf("\tKEY\t-->\tVALUE\n");
    for (long i = 0; i < TABLE_SIZE; i++)
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

__global__ void parallel_insert(long *d_input, long *d_hm)
{
    /*
    To insert an array of data to HashTable all at once using GPU
    */
    long input_idx = blockIdx.x * INPUT_THREDS_NUM + threadIdx.x;
    long table_idx = hash(d_input[input_idx]);
    while (table_idx < TABLE_SIZE && d_hm[table_idx] != NULL_DATA)
        table_idx++;
    if (table_idx == TABLE_SIZE)
    {
        printf("No More Space for data - %li !\n", d_input[input_idx]);
        return;
    }
    else
        d_hm[table_idx] = d_input[input_idx];
}

__global__ void parallel_delete(long *d_del, long* d_hm){
    long del_idx = blockIdx.x*DELETE_THREDS_NUM + threadIdx.x;
    int hm_idx = hash(d_del[del_idx]);
    while(hm_idx < TABLE_SIZE && d_hm[hm_idx] != NULL_DATA){
        if(d_hm[hm_idx] == d_del[del_idx]){
            d_hm[hm_idx] = DELETED_DATA;
            return;
        }
    }
    printf("Not Found!\n");
}

int main(long argc, char **argv)
{

    // Variables in CPU
    printf("Initialising Hash-Table....\n");
    long *h_input = (long *)malloc(INPUT_ARRAY_BYTES); // Input Array
    long *h_delete = (long *)malloc(DELETE_ARRAY_BYTES); // Input Array
    long *h_hm = (long *)malloc(TABLE_BYTES); // Hashtable in Host Memory
    initializeHashTable(h_hm);

    // Variables in GPU
    long *d_input;
    long *d_delete;
    long *d_hm; // Hash Table in Device/GPU Memory
    cudaMalloc((void **)&d_input, INPUT_ARRAY_BYTES);
    cudaMalloc((void **)&d_delete, DELETE_ARRAY_BYTES);
    cudaMalloc((void **)&d_hm, TABLE_BYTES);

    for(long i = 0; i < INPUT_ARRAY_SIZE; i++){
        h_input[i] = i;
    }
    cudaMemcpy(d_input, h_input, INPUT_ARRAY_BYTES, cudaMemcpyHostToDevice);
    parallel_insert<<<INPUT_BLOCKS_NUM, INPUT_THREDS_NUM>>>(d_input, d_hm);
    cudaMemcpy(h_hm, d_hm, TABLE_BYTES, cudaMemcpyDeviceToHost);

    // Printing the resulting HashTable
    printf("The Resulting Hash Table is saved in 'output1.txt' file.\n");
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