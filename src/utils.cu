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
    int N_v, int nf1, int nf2, int nf3, int plane, int pirange){
  /*
  do not use privitization due to sparsity. 
  histogram one plane by one plane (limitation of memory) ++++ sorted
  */
  int idx;
  int bin_x, bin_y, bin_z;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, pirange));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, pirange));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, pirange));
    /// 2d or somehow 3d partical
    if(bin_z==plane){
      int bindex = bin_y * nf1 + bin_x;
      int old = atomicAdd(&histo_count[bindex],1);
      sortidx_bin[idx] = old;
    }
  }
}

void part_histogram_3d_sparse_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int plane, int pirange){
  int blocksize = 256;
  part_histogram_3d_sparse<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,plane,pirange);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void histogram_3d_sparse(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count,
    int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, int nhive_x, int nhive_y, int nhive_z, int pirange){
  /*
  do not use privitization due to sparsity. 
  */
  // found the reason
  
  int idx;
  int bin_x, bin_y, bin_z;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, pirange));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, pirange));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, pirange));
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

__global__ void final_hive_plane_histo(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count,
    int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, 
    int nhive_x, int nhive_y, int nhive_z, int plane, int pirange){
  int idx;
  int bin_x, bin_y, bin_z;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, pirange));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, pirange));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, pirange));
    // if(idx<100)printf("----------%.6lf,%.6lf,%.6lf %d, %d, %d\n",x[idx],y[idx],z[idx], bin_x, bin_y, bin_z);
    int hive_x = bin_x / hivesize_x;
    int hive_y = bin_y / hivesize_y;
    // int hive_z = bin_z / hivesize_z;
    // cube_id
    if(bin_z >= plane){ // constant nf3->nhive
      histo_idx = hive_x + hive_y * nhive_x;
      histo_idx *= hivesize_x * hivesize_y * hivesize_z;
      histo_idx += bin_x % hivesize_x + (bin_y % hivesize_y) * hivesize_x + (bin_z - plane) * hivesize_x * hivesize_y;
      // printf("%d,%d,%d\n",hive_x,hive_y,hive_z);
      int old = atomicAdd(&histo_count[histo_idx],1);
      sortidx_bin[idx] = old;
    }
  }
}
__global__ void final_hive_mapping_gather(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, 
    int nhive_x, int nhive_y, int nhive_z, int plane, int total, int pirange){
  int idx;
  PCS x1, y1, z1;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    x1 = SHIFT_RESCALE(x[idx], nf1, pirange);
		y1 = SHIFT_RESCALE(y[idx], nf2, pirange);
		z1 = SHIFT_RESCALE(z[idx], nf3, pirange);
    int bin_x = floor(x1);
    int bin_y = floor(y1);
    int bin_z = floor(z1);
    int hive_x = bin_x / hivesize_x;
    int hive_y = bin_y / hivesize_y;
    // int hive_z = bin_z / hivesize_z;
    // if(abs(x[idx]-0.152601)<0.0001)printf("---%d,%d,%d\n",hive_x,hive_y,hive_z);
    if(bin_z>=plane){
      histo_idx = hive_x + hive_y * nhive_x;
      histo_idx *= hivesize_x * hivesize_y * hivesize_z;
      histo_idx += bin_x % hivesize_x + (bin_y % hivesize_y) * hivesize_x + (bin_z - plane) * hivesize_x * hivesize_y;

      int loc = N_v - (total - sortidx_bin[idx]-histo_count[histo_idx]);
      // if(abs(x[idx]-0.152601)<0.0001)printf("-------loc %d\n",loc);
      x_out[loc] = x1;
      y_out[loc] = y1;
      z_out[loc] = z1;
      c_out[loc] = c[idx];
    }
  }
}

void final_hive_plane_bin_mapping(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int *hive_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int* nhive, int pirange){
  int plane = nf3 - 8;
  int blocksize = 256;
  // printf("1111\n");
  final_hive_plane_histo<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0], hivesize[1], hivesize[2], nhive[0], nhive[1], nhive[2], plane, pirange);
  checkCudaErrors(cudaDeviceSynchronize());

  prefix_scan(histo_count,histo_count,nhive[0]*nhive[1]*hivesize[0]*hivesize[1]*hivesize[2]+1,0);
  int total;
  checkCudaErrors(cudaMemcpy(&total,histo_count+nhive[0]*nhive[1]*hivesize[0]*hivesize[1]*hivesize[2],sizeof(int),cudaMemcpyDeviceToHost));
  // count
  counting_hive_invoker(hive_count,histo_count,nhive[0]*nhive[1]+1,hivesize[0]*hivesize[1]*hivesize[2],N_v-total);
  final_hive_mapping_gather<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,c,x_out,y_out,z_out,c_out,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0],hivesize[1],hivesize[2],nhive[0],nhive[1],nhive[2],plane,total,pirange);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void histogram_3d_ignore_inbinorder(PCS *x, PCS *y, PCS *z, int *sortidx_hive, int *histo_count, int N_v,
    int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z,
    int nhive_x, int nhive_y, int nhive_z, int pirange){
  
  int idx;
  int bin_x, bin_y, bin_z;
  unsigned long int hive_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, pirange));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, pirange));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, pirange));
    int hive_x = bin_x / hivesize_x;
    int hive_y = bin_y / hivesize_y;
    int hive_z = bin_z / hivesize_z;

    hive_idx = hive_x + hive_y * nhive_x + hive_z * nhive_x * nhive_y;
    int old = atomicAdd(&histo_count[hive_idx],1);
    sortidx_hive[idx] = old;
  }
}

void histogram_3d_ignore_inbinorder_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_hive, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize,
    int *nhive, int pirange){
  int blocksize = 256;
  histogram_3d_ignore_inbinorder<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,sortidx_hive,histo_count,N_v,nf1,nf2,nf3,hivesize[0],hivesize[1],hivesize[2],nhive[0],nhive[1],nhive[2],pirange);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void mapping_based_gather_3d_ignore_ibo(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int nhive_x, int nhive_y, int nhive_z, int hivesize_x, int hivesize_y, int hivesize_z, int pirange){
  
  int idx;
  int bin_x, bin_y, bin_z;
  unsigned long int hive_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, pirange));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, pirange));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, pirange));
    int hive_x = bin_x / hivesize_x;
    int hive_y = bin_y / hivesize_y;
    int hive_z = bin_z / hivesize_z;

    hive_idx = hive_x + hive_y * nhive_x + hive_z * nhive_x * nhive_y;

    int loc = histo_count[hive_idx] + sortidx_bin[idx];
    x_out[loc] = x[idx];
    y_out[loc] = y[idx];
    z_out[loc] = z[idx];
    c_out[loc] = c[idx];
  }
}

void mapping_based_gather_3d_ignore_ibo_invoker(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int *nhive, int pirange){
  int blocksize = 256;
  mapping_based_gather_3d_ignore_ibo<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,c,x_out,y_out,z_out,c_out,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,nhive[0],nhive[1],nhive[2],hivesize[0],hivesize[1],hivesize[2],pirange);
  checkCudaErrors(cudaDeviceSynchronize());
}


__global__ void part_histogram_3d_cube(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count,
    int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, 
    int nhive_x, int nhive_y, int nhive_z, int cube_id, int cube_z, int pirange){
  /*
  do not use privitization due to sparsity. 
  */
  // found the reason
  
  int idx;
  int bin_x, bin_y, bin_z;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, pirange));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, pirange));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, pirange));
    int hive_x = bin_x / hivesize_x;
    int hive_y = bin_y / hivesize_y;
    int hive_z = bin_z / hivesize_z;
    // cube_id
    if(bin_z/cube_z==cube_id){ // constant nf3->nhive
      histo_idx = hive_x + hive_y * nhive_x + (hive_z-cube_id*cube_z/hivesize_z) * nhive_x * nhive_y;
      histo_idx *= hivesize_x * hivesize_y * hivesize_z;
      histo_idx += bin_x % hivesize_x + (bin_y % hivesize_y) * hivesize_x + (bin_z % hivesize_z) * hivesize_x * hivesize_y;
      // printf("%d,%d,%d\n",hive_x,hive_y,hive_z);
      int old = atomicAdd(&histo_count[histo_idx],1);
      sortidx_bin[idx] = old;
    }
  }
}

void histogram_3d_cube_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count,
    int N_v, int nf1, int nf2, int nf3, int *hivesize, int* nhive, int cube_id, int cube_z, int pirange){
  int blocksize = 512;
  part_histogram_3d_cube<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0], hivesize[1], hivesize[2], nhive[0], nhive[1], nhive[2], cube_id, cube_z, pirange);
  checkCudaErrors(cudaDeviceSynchronize());
}

void histogram_3d_sparse_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int* nhive, int pirange){
  int blocksize = 512;
  histogram_3d_sparse<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0], hivesize[1], hivesize[2], nhive[0], nhive[1], nhive[2], pirange);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void part_mapping_based_gather_3d(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int2 *se_loc, int N_v, int nf1, int nf2, int nf3, int plane, int init_scan_value, int pirange){
  int idx;
  int temp1, temp2, temp3;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    temp1 = floor(SHIFT_RESCALE(x[idx], nf1, pirange));
		temp2 = floor(SHIFT_RESCALE(y[idx], nf2, pirange));
		temp3 = floor(SHIFT_RESCALE(z[idx], nf3, pirange));
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
    int *sortidx_bin, int *histo_count, int2 *se_loc, int N_v, int nf1, int nf2, int nf3, int plane, int init_scan_value, int pirange){
  int blocksize = 256;
  part_mapping_based_gather_3d<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,c,x_out,y_out,z_out,c_out,sortidx_bin,histo_count,se_loc,N_v,nf1,nf2,nf3,plane,init_scan_value,pirange);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void part_mapping_based_gather_3d(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, 
    int nhive_x, int nhive_y, int nhive_z, int cube_id, int cube_z, int init_scan_value, int pirange){
  // replace issue
  int idx;
  PCS x1, y1, z1;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    x1 = SHIFT_RESCALE(x[idx], nf1, pirange);
		y1 = SHIFT_RESCALE(y[idx], nf2, pirange);
		z1 = SHIFT_RESCALE(z[idx], nf3, pirange);
    int bin_x = floor(x1);
    int bin_y = floor(y1);
    int bin_z = floor(z1);
    int hive_x = bin_x / hivesize_x;
    int hive_y = bin_y / hivesize_y;
    int hive_z = bin_z / hivesize_z;
    // if(abs(x[idx]-0.152601)<0.0001)printf("---%d,%d,%d\n",hive_x,hive_y,hive_z);
    if(bin_z/cube_z==cube_id){
      histo_idx = hive_x + hive_y * nhive_x + (hive_z-cube_id*cube_z/hivesize_z) * nhive_x * nhive_y;
      histo_idx *= hivesize_x * hivesize_y * hivesize_z;
      histo_idx += bin_x % hivesize_x + (bin_y % hivesize_y) * hivesize_x + (bin_z % hivesize_z) * hivesize_x * hivesize_y;

      int loc = sortidx_bin[idx]+histo_count[histo_idx]+init_scan_value;
      // if(abs(x[idx]-0.152601)<0.0001)printf("-------loc %d\n",loc);
      // if(loc==99971)printf("%lf %lf %lf %d %d %d %d\n",x1,y1,z1,bin_x,bin_y,bin_z,idx);
      x_out[loc] = x[idx];
      y_out[loc] = y[idx];
      z_out[loc] = z[idx];
      c_out[loc] = c[idx];
    }
  }
}

void part_mapping_based_gather_3d_invoker(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int *nhive, int cube_id, int cube_z, 
    int init_scan_value, int pirange){
  int blocksize = 256;
  part_mapping_based_gather_3d<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,c,x_out,y_out,z_out,c_out,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0],hivesize[1],hivesize[2],nhive[0],nhive[1],nhive[2],cube_id,cube_z,init_scan_value,pirange);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void mapping_based_gather_3d(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, int nhive_x, int nhive_y, int nhive_z, int pirange){
  int idx;
  int bin_x, bin_y, bin_z;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    bin_x = floor(SHIFT_RESCALE(x[idx], nf1, pirange));
		bin_y = floor(SHIFT_RESCALE(y[idx], nf2, pirange));
		bin_z = floor(SHIFT_RESCALE(z[idx], nf3, pirange));

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
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int hivesize_x, int hivesize_y, int hivesize_z, int nhive_x, int nhive_y, int nhive_z, int pirange){
  int idx;
  PCS x1, y1, z1;
  unsigned long int histo_idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N_v; idx+=blockDim.x*gridDim.x){
    // get bin index
    x1 = SHIFT_RESCALE(x[idx], nf1, pirange);
		y1 = SHIFT_RESCALE(y[idx], nf2, pirange);
		z1 = SHIFT_RESCALE(z[idx], nf3, pirange);
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
    int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int* nhive, int method, int pirange){
  int blocksize = 256;
  if(method==2||method==5||method==6)
  mapping_based_gather_3d<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,c,x_out,y_out,z_out,c_out,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0], hivesize[1], hivesize[2], nhive[0], nhive[1], nhive[2], pirange);
  if(method==3||method==4)mapping_based_gather_3d_replace<<<(N_v-1)/blocksize+1,blocksize>>>(x,y,z,c,x_out,y_out,z_out,c_out,sortidx_bin,histo_count,N_v,nf1,nf2,nf3,hivesize[0], hivesize[1], hivesize[2], nhive[0], nhive[1], nhive[2], pirange);
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
  nf = (int)nfd; // even number
  nf = nf%2 + nf;
  //printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  // catch too small nf, and nan or +-inf, otherwise spread fails...
  if (nf<2*spopts.kw) nf=2*spopts.kw;
  h = 2*PI / nf;                            // upsampled grid spacing
  gam = (PCS)nf / (2.0*spopts.upsampfac*Ssafe);  // x scale fac to x'
}

void taylor_series_approx_factors(PCS *c0, PCS *c1, PCS *c2, PCS *c3, double beta, int N){
  for(int i=0; i<N; i++){
    double x = i / (double) N;
    c0[i] = exp(beta*sqrt(1-x*x));
    c1[i] = -beta*x*c0[i] / sqrt(1-x*x); //power
    c2[i] = - beta*(beta*x*x*pow((1-x*x),1.5) + x*x -1)*c0[i] / pow((1-x*x),1.5)/(x*x-1) /2.0; //some error here
    c3[i] = beta*x*(3*beta*pow((1-x*x),2.5)+beta*beta*pow(x,8)-3*beta*beta*pow(x,6)+(3*beta*beta-3)*pow(x,4)+(6-beta*beta)*x*x-3)*c0[i]/
            pow((1-x*x),2.5)/pow((x*x-1),2) /6.0;
  }
}

void taylor_series_approx_factors(PCS *c0, double beta, int N, int N_order, int func_type){
  for(int i=0; i<N; i++){
    double x = i / (double) N;
    switch (N_order)
    {
      case 9:
        c0[i*N_order+8] = beta*(pow(beta*x,7)*x/pow(x*x-1,4) -28*pow(beta*x,6)*x*x/pow(1-x*x,4.5) -28*pow(beta*x,6)/pow(1-x*x,3.5) -378*pow(beta,5)*pow(x,8)/pow(x*x-1,5) +588*pow(beta*x,5)*x/pow(x*x-1,4) -210*pow(beta*x,4)*beta/pow(x*x-1,3) -3150*pow(beta,4)*pow(x,8)/pow(1-x*x,5.5) -5880*pow(beta*x,4)*x*x/pow(1-x*x,4.5) -
                      3150*pow(beta*x,4)/pow(1-x*x,3.5) -420*pow(beta,4)*x*x/pow(1-x*x,2.5) +17325*pow(beta,3)*pow(x,8)/pow(x*x-1,6) -35280*pow(beta,3)*pow(x,6)/pow(x*x-1,5) + 22050*pow(beta*x,3)*x/pow(x*x-1,4) -4200*pow(beta*x,2)*beta/pow(x*x-1,3) +105*pow(beta,3)/pow(x*x-1,2) -62370*beta*beta*pow(x,8)/pow(1-x*x,6.5) -
                      132300*beta*beta*pow(x,6)/pow(1-x*x,5.5) -88200*beta*beta*pow(x,4)/pow(1-x*x,4.5) -18900*pow(beta*x,2)/pow(1-x*x,3.5) - 630*beta*beta/pow(1-x*x,2.5) -135135*beta*pow(x,8)/pow(x*x-1,7) +291060*beta*pow(x,6)/pow(x*x-1,6) -198450*beta*pow(x,4)/pow(x*x-1,5) +44100*beta*x*x/pow(x*x-1,4) -1575*beta/pow(x*x-1,3) -
                      135135*pow(x,8)/pow(1-x*x,7.5) -291060*pow(x,6)/pow(1-x*x,6.5) -198450*pow(x,4)/pow(1-x*x,5.5) -44100*x*x/pow(1-x*x,4.5) -1575/pow(1-x*x,3.5))*c0[i*N_order]/40320.0;
      case 8:
        c0[i*N_order+7] = beta*x*(-pow(beta*x,6)/pow(1-x*x,3.5) + 21*pow(beta*x,5)*x/pow(x*x-1,4) - 21*pow(beta*x,4)*beta/pow(x*x-1,3) -210*pow(beta*x,4)*x*x/pow(1-x*x,4.5) -315*pow(beta*x,4)/pow(1-x*x,3.5) -105*pow(beta,4)*x*x/pow(1-x*x,2.5) -1260*pow(beta,3)*pow(x,6)/pow(x*x-1,5) +
                      2205*pow(beta*x,3)*x/pow(x*x-1,4) -1050*pow(beta*x,2)*beta/pow(x*x-1,3) +105*pow(beta,3)/pow(x*x-1,2) -4725*beta*beta*pow(x,6)/pow(1-x*x,5.5) - 8820*pow(beta*x,2)*x*x/pow(1-x*x,4.5) -4725*pow(beta*x,2)/pow(1-x*x,3.5) -630*beta*beta/pow(1-x*x,2.5) +
                      10395*beta*pow(x,6)/pow(x*x-1,6) - 19845*beta*pow(x,4)/pow(x*x-1,5) + 11025*beta*x*x/pow(x*x-1,4) -1575*beta/(x*x-1,3) -10395*pow(x,6)/pow(1-x*x,6.5) -19845*pow(x,4)/pow(1-x*x,5.5) - 11025*x*x/pow(1-x*x,4.5) -1575/(1-x*x,3.5))*c0[i*N_order]/5040.0;
      case 7:
        c0[i*N_order+6] = beta*(-x*pow(beta*x,5)/pow(x*x-1,3) -15*pow(beta*x,4)*x*x/pow(1-x*x,3.5) - 15*pow(beta*x,4)/pow(1-x*x,2.5) +105*pow(beta,3)*pow(x,6)/pow(x*x-1,4) -150*pow(beta*x,3)*x/pow(x*x-1,3) +45*pow(beta,3)*x*x/pow(x*x-1,2) -420*beta*beta*pow(x,6)/pow(1-x*x,4.5) -
                      675*beta*beta*pow(x,4)/pow(1-x*x,3.5) -270*beta*beta*x*x/pow(1-x*x,2.5) -15*beta*beta/pow(1-x*x,1.5) - 945*beta*pow(x,6)/pow(x*x-1,5) +1575*beta*pow(x,4)/pow(x*x-1,4) -675*beta*x*x/pow(x*x-1,3) + 45*beta/pow(x*x-1,2) -
                      945*pow(x,6)/pow(1-x*x,5.5) -1575*pow(x,4)/pow(1-x*x,4.5) - 675*x*x/pow(1-x*x,3.5) -45/pow(1-x*x,2.5))*c0[i*N_order]/720.0;
      case 6:
        c0[i*N_order+5] = beta*x*(-pow(x*beta,4) / pow((1-x*x),2.5) -10*pow(beta,3)*pow(x,4)/pow(x*x-1,3) +10*pow(beta,3)*x*x/pow(x*x-1,2) -45*beta*beta*pow(x,4)/pow(1-x*x,3.5) -60*beta*beta*x*x/pow(1-x*x,2.5) -15*beta*beta/pow(1-x*x,1.5) +
                      105*beta*pow(x,4)/pow(x*x-1,4) - 150*beta*x*x/pow(x*x-1,3) + 45*beta/pow(x*x-1,2) - 105*pow(x,4)/pow(1-x*x,4.5) -150*x*x/pow(1-x*x,3.5) -45/pow(1-x*x,2.5) )*c0[i*N_order]/120.0;
      case 5:
        c0[i*N_order+4] = beta*(6*beta*beta*pow(x,10)+(-24*beta*beta-12)*pow(x,8)+pow((1-x*x),3.5)*(beta*beta*beta*pow(x,6)-pow(beta,3)*pow(x,4)-12*beta*x*x-3*beta)+
                      (36*beta*beta+33)*pow(x,6)+(-24*beta*beta-27)*pow(x,4)+(6*beta*beta+3)*x*x+3)*c0[i*N_order] /
                      pow((1-x*x),3.5)/pow((x*x-1),3)/24.0;
      case 4:
        c0[i*N_order+3] = beta*x*(3*beta*pow((1-x*x),2.5)+beta*beta*pow(x,8)-3*beta*beta*pow(x,6)+(3*beta*beta-3)*pow(x,4)+(6-beta*beta)*x*x-3)*c0[i*N_order]/
            pow((1-x*x),2.5)/pow((x*x-1),2) /6.0;
      case 3:
        c0[i*N_order+2] = - beta*(beta*x*x*pow((1-x*x),1.5) + x*x -1)*c0[i*N_order] / pow((1-x*x),1.5)/(x*x-1) /2.0;
      case 2:
        c0[i*N_order+1] =  -beta*x*c0[i*N_order] / sqrt(1-x*x);
      case 1:
        c0[i*N_order] = exp(beta*(sqrt(1-x*x)-func_type));
      
      default:
        break;
    }
    // c0[i*N_order+9] = beta*x*(-pow(beta*x,8)/pow(1-x*x,4.5) -36*pow(beta*x,7)*x/pow(x*x-1,5) +36*pow(beta*x,6)*beta/pow(x*x-1,4) -630*pow(beta*x,6)*x*x/pow(1-x*x,5.5) -1008*pow(beta*x,6)/pow(1-x*x,4.5) -378*beta*beta*pow(beta*x,4)/pow(1-x*x,3.5) +6930*pow(beta,5)*pow(x,8)/pow(x*x-1,6) -13608*pow(beta*x,5)*x/pow(x*x-1,5) +
    //                   7938*pow(beta*x,4)*beta/pow(x*x-1,4) -1260*pow(beta,5)*x*x/pow(x*x-1,3) -51975*pow(beta,4)*pow(x,8)/pow(1-x*x,6.5) -113400*pow(beta*x,4)*x*x/pow(1-x*x,5.5) - 79380*pow(beta*x,4)/pow(1-x*x,4.5) -18900*pow(beta*x,2)*beta*beta/pow(1-x*x,3.5) -945*pow(beta,4)/pow(1-x*x,2.5) -270270*pow(beta,4)*pow(x,8)/pow(x*x-1,7) +
    //                   623700*pow(beta,3)*pow(x,6)/pow(x*x-1,6) -476280*pow(beta*x,3)*x/pow(x*x-1,5) +132300*pow(beta*x,2)*beta/pow(x*x-1,4) - 9450*pow(beta,3)/pow(x*x-1,3) -945945*beta*beta*pow(x,8)/pow(1-x*x,7.5) -2245320*beta*beta*pow(x,6)/pow(1-x*x,6.5) - 1786050*beta*beta*pow(x,4)/pow(1-x*x,5.5) -529200*pow(beta*x,2)/pow(1-x*x,4.5) -
    //                   42525*beta*beta/pow(1-x*x,3.5) +2027025*beta*pow(x,8)/pow(x*x-1,8) -4864860*beta*pow(x,6)/pow(x*x-1,7) + 3929310*beta*pow(x,4) -1190700*beta*x*x/pow(x*x-1,5) +99225*beta/pow(x*x-1,4) - 2027025*pow(x,8)/pow(1-x*x,8.5) -4864860*pow(x,6)/pow(1-x*x,7.5) -3929310*pow(x,4)/pow(1-x*x,6.5) - 1190700*x*x/pow(1-x*x,5.5) -99225/pow(1-x*x,4.5) )*c0[i*N_order]/362880.0; 
    // c0[i*N_order+10] = 21992.1238;
  }
}