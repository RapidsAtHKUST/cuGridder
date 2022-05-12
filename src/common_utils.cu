/* common utility functions
  1. GPU info
  2. GPU memory status
  3. next2357
*/

#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <iostream>
#include <stdio.h>
//#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "datatype.h"
void GPU_info()
{

    printf("Starting... \n");
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
        printf("There is no available device that support CUDA\n");
    else
        printf("Detected %d CUDA capable device(s)\n", deviceCount);

    int dev, driverVersion = 0, runtimeVersion = 0;

    dev = 0;

    printf("Input the device index:");
    std::cin >> dev;
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: %s\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version / Runtime Version       %d.%d / %d.%d\n",
           driverVersion / 1000, driverVersion % 1000, runtimeVersion / 1000, runtimeVersion % 1000);
    printf("CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Total amount of global memory:              %.2f MBytes\n", (float)deviceProp.totalGlobalMem / (pow(1024.0, 2)));
    printf("GPU clock rate:                             %.0f MHz\n", deviceProp.clockRate * 1e-3f);
    printf("Memory clock rate:                          %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
    printf("Memory Bus Width:                           %d-bit\n", deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize)
    {
        printf("L2 Cache Size:                          %d bytes\n", deviceProp.l2CacheSize);
    }
    printf("Total amount of constant memory:            %lu bytes\n", deviceProp.totalConstMem);
    printf("Total amount of shared memory per block:    %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("Warp size:                                  %d\n", deviceProp.warpSize);
    printf("Number of multiprocessors:                  %d\n", deviceProp.multiProcessorCount);
    printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    //printf("Maximum number of blocks per multiprocessor: %d\n",deviceProp.maxBlocksPerMultiProcessor);
    printf("Maximum number of thread per block:          %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum sizes of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Maximum sizes of each dimension of a grid:  %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("Maximum memory pitch:                       %lu bytes\n", deviceProp.memPitch);
}

int next235beven(int n, int b)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth") and is a multiple of b (b is a number that the only prime
// factors are 2,3,5). Adapted from fortran in hellskitchen. Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
// added condition about b Melody 05/31/20
{
    if (n <= 2)
        return 2;
    if (n % 2 == 1)
        n += 1;        // even
    int nplus = n - 2; // to cancel out the +=2 at start of loop
    int numdiv = 2;    // a dummy that is >1
    while ((numdiv > 1) || (nplus % b != 0))
    {
        nplus += 2; // stays even
        numdiv = nplus;
        while (numdiv % 2 == 0)
            numdiv /= 2; // remove all factors of 2,3,5...
        while (numdiv % 3 == 0)
            numdiv /= 3;
        while (numdiv % 5 == 0)
            numdiv /= 5;
    }
    return nplus;
}

void show_mem_usage()
{
    // show memory usage of GPU

    size_t free_byte;

    size_t total_byte;

    checkCudaErrors(cudaMemGetInfo(&free_byte, &total_byte));

    double free_db = (double)free_byte;

    double total_db = (double)total_byte;

    double used_db = total_db - free_db;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
           used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

__global__ void counting_hive(int *hive_count, int *histo_count, unsigned long int M, int hivesize)
{
    unsigned int idx;
    for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x)
    {
        hive_count[idx] = histo_count[idx * hivesize];
        // printf("%d, %d\n",idx, hive_count[idx]);
    }
}

void counting_hive_invoker(int *hive_count, int *histo_count, unsigned long int hive_count_size, int hivesize)
{
    int blocksize = 256;
    counting_hive<<<(hive_count_size - 1) / blocksize + 1, blocksize>>>(hive_count, histo_count, hive_count_size, hivesize);
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void counting_hive(int *hive_count, int *histo_count, unsigned long int M, int hivesize, int initial)
{
    unsigned int idx;
    for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x)
    {
        hive_count[idx] = histo_count[idx * hivesize] + initial;
    }
}

void counting_hive_invoker(int *hive_count, int *histo_count, unsigned long int hive_count_size, int hivesize, int initial)
{
    int blocksize = 256;
    counting_hive<<<(hive_count_size - 1) / blocksize + 1, blocksize>>>(hive_count, histo_count, hive_count_size, hivesize, initial);
    checkCudaErrors(cudaDeviceSynchronize());
}

void prefix_scan(int *d_arr, int *d_res, unsigned long long int n, int flag)
{
    /*
    n - number of elements
    flag - 1 inclusive, 0 exclusive
    thrust::inclusive_scan(d_arr, d_arr + n, d_res);
  */
    thrust::device_ptr<int> d_ptr(d_arr); // not convert
    thrust::device_ptr<int> d_result(d_res);

    if (flag)
        thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    else
        thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
}