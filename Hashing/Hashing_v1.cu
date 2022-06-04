/*
CUDA Implementation of HashTable:
-> OPEN ADDRESSING is used to handle collisions
-> Some simple non-parallel functions are provided for insertion, deletion, finding single elements in HashTable
-> Kernels are defined to Insert or Delete an Array of data simultaneously to the HashTable
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define ARRAY_SIZE 10000 // Size of Input Data
#define ARRAY_BYTES (ARRAY_SIZE * sizeof(long))
#define TABLE_SIZE 100000 // Size of Hash Table
#define TABLE_BYTES (TABLE_SIZE * sizeof(long))
#define NUM_THREADS1 1000 // NUM_THREADS1 <= ARRAY_SIZE && ARRAY_SIZE % NUM_THREADS1 == 0
#define DEL_ARRAY_SIZE 1000 // Array of data to be deleted
#define DEL_ARRAY_BYTES DEL_ARRAY_SIZE*sizeof(long)
#define NUM_THREADS2 1000 // NUM_THREADS2 <= DEL_ARRAY_SIZE && DEL_ARRAY_SIZE % NUM_THREADS2 == 0
#define NULL_DATA NULL
#define DELETED_DATA LONG_MIN

__device__ __host__ long hash(long n)
{
    /*
    A simple hash function to be used in our HashTable
    */
    return (n * n) % TABLE_SIZE;
}

void hm_init(long *hm)
{
    for (long i = 0; i < TABLE_SIZE; i++)
    {
        hm[i] = NULL_DATA;
    }
}

void hm_print(long *hm)
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

void hm_insert(long *hm, long i)
{
    /*
    To insert a single element to HashTable (Works in CPU)
    */
    long idx = hash(i);
    while (idx < TABLE_SIZE && hm[idx] != NULL_DATA && hm[idx] != DELETED_DATA)
        idx++;
    if (idx == TABLE_SIZE)
    {
        printf("No More Space for data - %li !\n", i);
        return;
    }
    else
        hm[idx] = i;
}

void hm_delete(long *hm, long data)
{
    /*
    To delete single data from HashTable (if available)
    */
    int idx = hash(data);
    while (idx < TABLE_SIZE && hm[idx] != data)
    {
        if (hm[data] == DELETED_DATA)
            continue;
        else if (hm[idx] == data)
        {
            hm[idx] = DELETED_DATA;
            return;
        }
        else if (hm[idx] == NULL_DATA)
        {
            printf("Not Found!\n");
            return;
        }
        idx++;
    }
    printf("Not Found!\n");
}

void hm_find(long *hm, long data)
{
    /*
    To find if 'data' is present in HashMap or Not
    */
    printf("Searching for (%li) in Hashable.....\n", data);
    long idx = hash(data);
    while (idx < TABLE_SIZE)
    {
        if (hm[idx] == data)
        {
            printf("Found at Index = %li\n", idx);
            return;
        }
        else if (hm[idx] == DELETED_DATA)
            idx++;
        else if (hm[idx] == NULL_DATA)
        {
            printf("Data Not Found!\n");
            return;
        }
    }
    printf("Data Not Found!\n");
}

__global__ void parallel_insert(long *d_input, long *d_hm)
{
    /*
    To insert an array of data to HashTable all at once using GPU
    */
    long input_idx = blockIdx.x * NUM_THREADS1 + threadIdx.x;
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
    long del_idx = blockIdx.x*NUM_THREADS2 + threadIdx.x;
    int hm_idx = hash(d_del[del_idx]);
    while(hm_idx < TABLE_SIZE && d_hm[hm_idx] != NULL_DATA){
        if(d_hm[hm_idx] == d_del[del_idx]){
            d_hm[hm_idx] = DELETED_DATA;
        }
    }
    printf("No Data to Delete!\n");
}

int main(long argc, char **argv)
{

    // Variables in CPU
    long *h_input = (long *)malloc(ARRAY_BYTES); // Input data to be stored
    for (long i = 0; i < ARRAY_SIZE; i++)
    {
        h_input[i] = i;
    }

    long *h_hm = (long *)malloc(TABLE_BYTES); // Hashtable in Host Memory
    hm_init(h_hm);

    // Variables in GPU
    long *d_input;
    long *d_hm; // Hash Table in Device/GPU Memory
    cudaMalloc((void **)&d_input, ARRAY_BYTES);
    cudaMalloc((void **)&d_hm, TABLE_BYTES);

    // Copying Input Data from Host to Device
    cudaMemcpy(d_input, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // Parallel Insertion of data to Hash Table
    parallel_insert<<<ARRAY_SIZE / NUM_THREADS1, NUM_THREADS1>>>(d_input, d_hm);

    // Copying Hash_Table Data from Device to Host
    cudaMemcpy(h_hm, d_hm, TABLE_BYTES, cudaMemcpyDeviceToHost);

    // Array of data to be deleted
    long*h_del = (long*)malloc(DEL_ARRAY_BYTES);
    for (long i = 0; i < DEL_ARRAY_SIZE; i++)
    {
        h_del[i] = i;
    }
    
    long* d_del;
    cudaMalloc((void **)&d_del, DEL_ARRAY_BYTES);
    cudaMemcpy(d_del, h_del, DEL_ARRAY_BYTES, cudaMemcpyHostToDevice);

    // Parallel Deletion of Data from Hash Table
    parallel_delete<<<DEL_ARRAY_SIZE/NUM_THREADS2, NUM_THREADS2>>>(d_del, d_hm);

    cudaMemcpy(h_hm, d_hm, TABLE_BYTES, cudaMemcpyDeviceToHost);

    // Testing hm_delete
    hm_delete(h_hm, 1);

    // Printing the resulting HashTable
    printf("The Resulting Hash Table is saved in 'output1.txt' file.\n");
    freopen("output1.txt", "w", stdout);
    hm_print(h_hm);

    free(h_input);
    free(h_del);
    free(h_hm);
    cudaFree(d_input);
    cudaFree(d_del);
    cudaFree(d_hm);
    return 0;
}
