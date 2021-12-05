#include <iostream>
#include <iomanip>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
#include "utils.h"

int main(int argc, char* argv[]){

    PCS *arr, *d_arr;
    int n = 10;
    arr = (PCS *) malloc (sizeof(PCS)*n);
    for(int i=0; i<n; i++){
        arr[i] = randm11()*0.5*PI; //convert to int for checking
        printf("%.3g ", arr[i]);
    }
    printf("\n");
    checkCudaErrors(cudaMalloc((void **)&d_arr, sizeof(PCS)*n));
    checkCudaErrors(cudaMemcpy(d_arr, arr, sizeof(PCS)*n, cudaMemcpyHostToDevice));

    /*-------------get_max_min test------------*/
    printf("Get max&min testing...\n");
    PCS max, min;
    get_max_min(max, min, d_arr, n);
    printf("max value is %.3g, min is %.3g\n", max, min);

    /*-------------prefix_scan test------------*/
    printf("Prefix scan testing...\n");
    prefix_scan(d_arr, d_arr, n, 0);
    checkCudaErrors(cudaMemcpy(arr, d_arr, sizeof(PCS)*n, cudaMemcpyDeviceToHost));
    for(int i=0; i<10; i++){
        printf("%.3g ", arr[i]);
    }
    printf("\n");

    free(arr);
    checkCudaErrors(cudaFree(d_arr));

    /*-------------sparse histogram test--------------*/
    printf("Histogram testing...\n");
    n = 10;
    PCS *x; PCS *y ;PCS *z;
    PCS *d_x, *d_y, *d_z;
    PCS *d_x_out, *d_y_out, *d_z_out;

    CUCPX *d_c, *d_c_out;
    int *sortidx_bin, *histo_count;
    int *h_sortidx_bin, *h_histo_count;
    int2 *se_loc;
    int2 *h_se_loc;
    x = (PCS *)malloc(sizeof(PCS)*n);
    y = (PCS *)malloc(sizeof(PCS)*n);
    z = (PCS *)malloc(sizeof(PCS)*n);
    CPX *c;
    c = (CPX *)malloc(sizeof(CPX)*n);
    for(int i=0; i<n; i++){
        x[i] = randm11()*0.5*PI;
        y[i] = randm11()*0.5*PI;
        z[i] = randm11()*0.5*PI;
        c[i].real(i/double(n));
        c[i].real(i);
        printf("%.3g,%.3g ", x[i],y[i]);
    }
    printf("\n");

    int nf1 = n*2;
    int nf2 = n*2;
    int nf3 = 2;
    h_sortidx_bin = (int *)malloc(sizeof(int)*n);
    h_histo_count = (int *)malloc(sizeof(int)*(nf1*nf2+1));
    h_se_loc = (int2 *)malloc(sizeof(int2)*n);
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(PCS)*n));
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(PCS)*n));
    checkCudaErrors(cudaMalloc((void **)&d_z,sizeof(PCS)*n));
    checkCudaErrors(cudaMalloc((void **)&d_x_out, sizeof(PCS)*n));
    checkCudaErrors(cudaMalloc((void **)&d_y_out, sizeof(PCS)*n));
    checkCudaErrors(cudaMalloc((void **)&d_z_out,sizeof(PCS)*n));
    checkCudaErrors(cudaMalloc((void **)&se_loc,sizeof(int2)*n));
    checkCudaErrors(cudaMalloc((void **)&d_c,sizeof(CUCPX)*n));
    checkCudaErrors(cudaMalloc((void **)&d_c_out,sizeof(CUCPX)*n));
    checkCudaErrors(cudaMalloc((void**)&sortidx_bin,sizeof(int)*n));
    checkCudaErrors(cudaMalloc((void**)&histo_count,sizeof(int)*(nf1*nf2+1)));
    
    checkCudaErrors(cudaMemcpy(d_x,x,sizeof(PCS)*n,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y,y,sizeof(PCS)*n,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_z,z,sizeof(PCS)*n,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_c,c,sizeof(CPX)*n,cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMemset(sortidx_bin,-1,sizeof(int)*n));
    checkCudaErrors(cudaMemset(histo_count,0,sizeof(int)*(nf1*nf2+1)));
    int init_scan_value = 0;
    for(int i=0; i<nf3; i++){
        part_histogram_3d_sparse_invoker(d_x,d_y,d_z,sortidx_bin,histo_count,n,nf1,nf2,nf3,i,1);
        prefix_scan(histo_count,histo_count,nf1*nf2+1,0);
        part_mapping_based_gather_3d_invoker(d_x,d_y,d_z,d_c,d_x_out,d_y_out,d_z_out,d_c_out,sortidx_bin,histo_count,se_loc,n,nf1,nf2,nf3,i,init_scan_value,plan->copts.pirange);
        int last_value;
        checkCudaErrors(cudaMemcpy(&last_value,histo_count+nf1*nf2,sizeof(int),cudaMemcpyDeviceToHost));
        init_scan_value += last_value;
        checkCudaErrors(cudaMemcpy(h_histo_count, histo_count, sizeof(int)*(nf1*nf2+1), cudaMemcpyDeviceToHost));
        for(int i=0; i<nf1*nf2+1; i++){
            printf("%d ",h_histo_count[i]);
        }
        printf("\n");
        checkCudaErrors(cudaMemset(histo_count,0,sizeof(int)*(nf1*nf2+1)));
    }
    checkCudaErrors(cudaMemcpy(x,d_x_out,sizeof(PCS)*n,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_se_loc,se_loc,sizeof(int2)*n,cudaMemcpyDeviceToHost));
    for(int i=0; i<n; i++){
        printf("%lf,%d,%d ",x[i],h_se_loc[i].x,h_se_loc[i].y);
    }
    printf("\n");



    checkCudaErrors(cudaMemcpy(h_sortidx_bin,sortidx_bin,sizeof(int)*n,cudaMemcpyDeviceToHost));
    // printf("sortidx_bin\n");
    // for(int i=0; i<n; i++){
    //     printf("%d ",h_sortidx_bin[i]);
    // }
    // printf("\n");

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_c);
    cudaFree(d_x_out);
    cudaFree(d_y_out);
    cudaFree(d_z_out);
    cudaFree(d_c_out);
    cudaFree(sortidx_bin);
    cudaFree(histo_count);
    cudaFree(se_loc);
    free(x);
    free(y);
    free(z);
    free(c);
    free(h_se_loc);
    free(h_histo_count);
    free(h_sortidx_bin);
    /*-------------GPU info test------------*/
    GPU_info();
    return 0;
}