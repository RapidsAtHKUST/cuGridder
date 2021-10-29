/*
  Utility functions
  1. prefix_scan
  2. get_max_min
  3. rescale
  4. shift_and_scale
  5. matrix transpose
*/

#include "utils.h"
#include "common_utils.h"
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
#include <cub/cub.cuh>
#include "datatype.h"
#include "curafft_plan.h"
#include "conv.h"

void prefix_scan(PCS *d_arr, PCS *d_res, int n, int flag)
{
  /*
    n - number of elements
    flag - 1 inclusive, 0 exclusive
    thrust::inclusive_scan(d_arr, d_arr + n, d_res);
  */
  thrust::device_ptr<PCS> d_ptr(d_arr); // not convert
  thrust::device_ptr<PCS> d_result(d_res);

  if (flag)
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
  else
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
}

void get_max_min(PCS &max, PCS &min, PCS *d_array, int n)
{
  /*
    Get the maximum and minimum of array by thrust
    d_array - array on device
    n - length of array
  */
  thrust::device_ptr<PCS> d_ptr = thrust::device_pointer_cast(d_array);
  max = *(thrust::max_element(d_ptr, d_ptr + n));

  min = *(thrust::min_element(d_ptr, d_ptr + n));
}

// __global__ part_histogram_3d(int N_v, int N_b, int scounter){
//   /*
//   N_v - number of points that need to map
//   N_b - number of bins
//   scounter - record start point
//   */
//   int idx;
//   for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
//     // get bin index
//     int bindex = 0;
    
//     // shared memory allocation and initate to 0

//     // 
//   }
// }

__global__ void part_histogram_3d_sparse(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count,
    int N_v, int nf1, int nf2, int nf3, int plane){
  /*
  do not use privitization due to sparsity. 
  histogram one plane by one plane (limitation of memory) ++++ sorted
  */
  int idx;
  int bin_x, bin_y, bin_z;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, 1));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, 1));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, 1));
    /// 2d or somehow 3d partical
    if(bin_z==plane){
      int bindex = bin_y * nf1 + bin_x;
      int old = atomicAdd(&histo_count[bindex],1);
      sortidx_bin[idx] = old;
    }
  }
}

void part_histogram_3d_sparse_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int plane){
  int blocksize = 256;
  part_histogram_3d_sparse<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,plane);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void histogram_3d_sparse(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count,
    int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, int nhive_x, int nhive_y, int nhive_z){
  /*
  do not use privitization due to sparsity. 
  */
  // found the reason
  
  int idx;
  int bin_x, bin_y, bin_z;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, 1));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, 1));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, 1));
    int hive_x = bin_x / hivesize_x;
    int hive_y = bin_y / hivesize_y;
    int hive_z = bin_z / hivesize_z;
    histo_idx = hive_x + hive_y * nhive_x + hive_z * nhive_x * nhive_y;
    histo_idx *= hivesize_x * hivesize_y * hivesize_z;
    histo_idx += bin_x % hivesize_x + (bin_y % hivesize_y) * hivesize_x + (bin_z % hivesize_z) * hivesize_x * hivesize_y;
    // printf("%d,%d,%d\n",hive_x,hive_y,hive_z);
    int old = atomicAdd(&histo_count[histo_idx],1);
    sortidx_bin[idx] = old;
    
  }
}

void histogram_3d_sparse_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int* nhive){
  int blocksize = 512;
  histogram_3d_sparse<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0], hivesize[1], hivesize[2], nhive[0], nhive[1], nhive[2]);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void part_mapping_based_gather_3d(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int2 *se_loc, int N_v, int nf1, int nf2, int nf3, int plane, int init_scan_value){
  int idx;
  int temp1, temp2, temp3;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    temp1 = floor(SHIFT_RESCALE(x[idx], nf1, 1));
		temp2 = floor(SHIFT_RESCALE(y[idx], nf2, 1));
		temp3 = floor(SHIFT_RESCALE(z[idx], nf3, 1));
    if(temp3==plane){
      int bindex = temp2 * nf1 + temp1;
      int start_loc = histo_count[bindex]+init_scan_value;
      int loc = sortidx_bin[idx]+start_loc;
      x_out[loc] = x[idx];
      y_out[loc] = y[idx];
      z_out[loc] = z[idx];
      c_out[loc] = c[idx];
      se_loc[loc].x = histo_count[bindex]+init_scan_value;
      se_loc[loc].y = histo_count[bindex+1]+init_scan_value;
    }
  }
}

void part_mapping_based_gather_3d_invoker(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int2 *se_loc, int N_v, int nf1, int nf2, int nf3, int plane, int init_scan_value){
  int blocksize = 256;
  part_mapping_based_gather_3d<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,c,x_out,y_out,z_out,c_out,sortidx_bin,histo_count,se_loc,N_v,nf1,nf2,nf3,plane,init_scan_value);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void mapping_based_gather_3d(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, int nhive_x, int nhive_y, int nhive_z){
  int idx;
  int bin_x, bin_y, bin_z;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, 1));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, 1));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, 1));

    int hive_x = bin_x / hivesize_x;
    int hive_y = bin_y / hivesize_y;
    int hive_z = bin_z / hivesize_z;
    // if(abs(x[idx]-0.152601)<0.0001)printf("---%d,%d,%d\n",hive_x,hive_y,hive_z);
    histo_idx = hive_x + hive_y * nhive_x + hive_z * nhive_x * nhive_y;
    histo_idx *= hivesize_x * hivesize_y * hivesize_z;
    histo_idx += bin_x % hivesize_x + (bin_y % hivesize_y) * hivesize_x + (bin_z % hivesize_z) * hivesize_x * hivesize_y;

    int loc = sortidx_bin[idx]+histo_count[histo_idx];
    // if(abs(x[idx]-0.152601)<0.0001)printf("-------loc %d\n",loc);
    x_out[loc] = x[idx];
    y_out[loc] = y[idx];
    z_out[loc] = z[idx];
    c_out[loc] = c[idx];
  }
}

__global__ void mapping_based_gather_3d_replace(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, int nhive_x, int nhive_y, int nhive_z){
  int idx;
  PCS x1, y1, z1;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    x1 = SHIFT_RESCALE(x[idx], nf1, 1);
		y1 = SHIFT_RESCALE(y[idx], nf2, 1);
		z1 = SHIFT_RESCALE(z[idx], nf3, 1);
    int bin_x = floor(x1);
    int bin_y = floor(y1);
    int bin_z = floor(z1);
    int hive_x = bin_x / hivesize_x;
    int hive_y = bin_y / hivesize_y;
    int hive_z = bin_z / hivesize_z;
    // if(abs(x[idx]-0.152601)<0.0001)printf("---%d,%d,%d\n",hive_x,hive_y,hive_z);
    histo_idx = hive_x + hive_y * nhive_x + hive_z * nhive_x * nhive_y;
    histo_idx *= hivesize_x * hivesize_y * hivesize_z;
    histo_idx += bin_x % hivesize_x + (bin_y % hivesize_y) * hivesize_x + (bin_z % hivesize_z) * hivesize_x * hivesize_y;

    int loc = sortidx_bin[idx]+histo_count[histo_idx];
    // if(abs(x[idx]-0.152601)<0.0001)printf("-------loc %d\n",loc);
    x_out[loc] = x1;
    y_out[loc] = y1;
    z_out[loc] = z1;
    c_out[loc] = c[idx];
  }
}

void mapping_based_gather_3d_invoker(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int* nhive, int method){
  int blocksize = 256;
  if(method==2)
  mapping_based_gather_3d<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,c,x_out,y_out,z_out,c_out,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0], hivesize[1], hivesize[2], nhive[0], nhive[1], nhive[2]);
  if(method==3)mapping_based_gather_3d_replace<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,c,x_out,y_out,z_out,c_out,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0], hivesize[1], hivesize[2], nhive[0], nhive[1], nhive[2]);
  checkCudaErrors(cudaDeviceSynchronize());
}

// real and complex array scaling
__global__ void rescaling_real(PCS *x, PCS scale_ratio, int N)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    x[idx] *= scale_ratio;
  }
}

__global__ void rescaling_complex(CUCPX *x, PCS scale_ratio, int N)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    x[idx].x *= scale_ratio;
    x[idx].y *= scale_ratio;
  }
}

void rescaling_real_invoker(PCS *d_x, PCS scale_ratio, int N)
{
  int blocksize = 512;
  rescaling_real<<<(N - 1) / blocksize + 1, blocksize>>>(d_x, scale_ratio, N);
  CHECK(cudaDeviceSynchronize());
}

void rescaling_complex_invoker(CUCPX *d_x, PCS scale_ratio, int N)
{
  int blocksize = 512;
  rescaling_complex<<<(N - 1) / blocksize + 1, blocksize>>>(d_x, scale_ratio, N);
  CHECK(cudaDeviceSynchronize());
}

__global__ void shift_and_scale(PCS i_center, PCS o_center, PCS gamma, PCS *d_u, PCS *d_x, int M, int N)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x)
  {
    d_u[idx] = (d_u[idx] - i_center) / gamma;
  }
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    d_x[idx] = (d_x[idx] - o_center) * gamma;
  }
}

void shift_and_scale_invoker(PCS i_center, PCS o_center, PCS gamma, PCS *d_u, PCS *d_x, int M, int N)
{
  // Specified for nu to nu fourier transform
  int blocksize = 512;
  shift_and_scale<<<(max(M, N) - 1) / blocksize + 1, blocksize>>>(i_center, o_center, gamma, d_u, d_x, M, N);
  CHECK(cudaDeviceSynchronize());
}

__global__ void transpose(PCS *odata, PCS *idata, int width, int height)
{
  //* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
  // refer https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
  __shared__ PCS block[BLOCKSIZE][BLOCKSIZE];

  // read the matrix tile into shared memory
  // load one element per thread from device memory (idata) and store it
  // in transposed order in block[][]
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //height
  unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //width
  if ((yIndex < width) && (xIndex < height))
  {
    unsigned int index_in = xIndex * width + yIndex;
    block[threadIdx.x][threadIdx.y] = idata[index_in];
  }

  // synchronise to ensure all writes to block[][] have completed
  __syncthreads();

  // write the transposed matrix tile to global memory (odata) in linear order
  xIndex = blockIdx.y * blockDim.x + threadIdx.x;
  yIndex = blockIdx.x * blockDim.y + threadIdx.y;
  if ((yIndex < height) && (xIndex < width))
  {
    unsigned int index_out = xIndex * height + yIndex;
    odata[index_out] = block[threadIdx.y][threadIdx.x];
  }
  // __syncthreads();
}

int matrix_transpose_invoker(PCS *d_arr, int width, int height)
{
  int ier = 0;
  int blocksize = BLOCKSIZE;
  dim3 block(blocksize, blocksize);
  dim3 grid((height - 1) / blocksize + 1, (width - 1) / blocksize + 1);
  PCS *temp_o;
  checkCudaErrors(cudaMalloc((void **)&temp_o, sizeof(PCS) * width * height));
  transpose<<<grid, block>>>(temp_o, d_arr, width, height);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(d_arr, temp_o, sizeof(PCS) * width * height, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(temp_o));
  return ier;
}

__global__ void matrix_elementwise_multiply(CUCPX *a, PCS *b, int N)
{
  int idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    a[idx].x = a[idx].x * b[idx];
    a[idx].y = a[idx].y * b[idx];
  }
}

int matrix_elementwise_multiply_invoker(CUCPX *a, PCS *b, int N)
{
  int ier = 0;
  int blocksize = 512;
  matrix_elementwise_multiply<<<(N - 1) / blocksize + 1, blocksize>>>(a, b, N);
  checkCudaErrors(cudaDeviceSynchronize());
  return ier;
}

__global__ void matrix_elementwise_divide(CUCPX *a, PCS *b, int N)
{
  int idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    a[idx].x = a[idx].x / b[idx];
    a[idx].y = a[idx].y / b[idx];
  }
}

int matrix_elementwise_divide_invoker(CUCPX *a, PCS *b, int N)
{
  int ier = 0;
  int blocksize = 512;
  matrix_elementwise_multiply<<<(N - 1) / blocksize + 1, blocksize>>>(a, b, N);
  checkCudaErrors(cudaDeviceSynchronize());
  return ier;
}

void set_nhg_w(PCS S, PCS X, conv_opts spopts,
		     int &nf, PCS &h, PCS &gam)
/* sets nf, h (upsampled grid spacing), and gamma (x_j rescaling factor),
   for type 3 only.
   Inputs:
   X and S are the xj and sk interval half-widths respectively.
   opts and spopts are the NUFFT and spreader opts strucs, respectively.
   Outputs:
   nf is the size of upsampled grid for a given single dimension.
   h is the grid spacing = 2pi/nf
   gam is the x rescale factor, ie x'_j = x_j/gam  (modulo shifts).
   Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
   New logic 6/12/17
*/
{
  int nss = spopts.kw + 1;      // since ns may be odd
  PCS Xsafe=X, Ssafe=S;              // may be tweaked locally
  if (X==0.0)                        // logic ensures XS>=1, handle X=0 a/o S=0
    if (S==0.0) {
      Xsafe=1.0;
      Ssafe=1.0;
    } else Xsafe = std::max(Xsafe, 1/S);
  else
    Ssafe = std::max(Ssafe, 1/X);
  // use the safe X and S...
  PCS nfd = 2.0*spopts.upsampfac*Ssafe*Xsafe/PI + nss;
  if (!isfinite(nfd)) nfd=0.0;                // use FLT to catch inf
  nf = (int)nfd;
  //printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  // catch too small nf, and nan or +-inf, otherwise spread fails...
  if (nf<2*spopts.kw) nf=2*spopts.kw;
  h = 2*PI / nf;                            // upsampled grid spacing
  gam = (PCS)nf / (2.0*spopts.upsampfac*Ssafe);  // x scale fac to x'
}