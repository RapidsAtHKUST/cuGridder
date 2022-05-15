#include <iostream>
#include <iomanip>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
#include "utils.h"

int main(int argc, char *argv[])
{

    PCS *arr, *d_arr;
    int n = 10;
    arr = (PCS *)malloc(sizeof(PCS) * n);
    for (int i = 0; i < n; i++)
    {
        arr[i] = randm11() * 0.5 * PI; //convert to int for checking
        printf("%.3g ", arr[i]);
    }
    printf("\n");
    checkCudaError(cudaMalloc((void **)&d_arr, sizeof(PCS) * n));
    checkCudaError(cudaMemcpy(d_arr, arr, sizeof(PCS) * n, cudaMemcpyHostToDevice));

    /*-------------get_max_min test------------*/
    printf("Get max&min testing...\n");
    PCS max, min;
    get_max_min(max, min, d_arr, n);
    printf("max value is %.3g, min is %.3g\n", max, min);

    /*-------------prefix_scan test------------*/
    printf("Prefix scan testing...\n");
    prefix_scan(d_arr, d_arr, n, 0);
    checkCudaError(cudaMemcpy(arr, d_arr, sizeof(PCS) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 10; i++)
    {
        printf("%.3g ", arr[i]);
    }
    printf("\n");

    free(arr);
    checkCudaError(cudaFree(d_arr));

    /*-------------GPU info test------------*/
    GPU_info();
    return 0;
}