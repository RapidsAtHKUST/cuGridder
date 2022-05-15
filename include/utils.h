#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
//#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "common_utils.h"
#include "datatype.h"
///contrib

// 0 correct  1 warning 2 error

#define checkCudaError(call)                                           \
    {                                                                   \
        const cudaError_t error = call;                                 \
        if (error != cudaSuccess)                                       \
        {                                                               \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,     \
            cudaGetErrorString(error));                                   \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    }

#define M_1_2PI 0.159154943091895336 // 1/2/pi for faster calculation
#define M_2PI 6.28318530717958648    // 2 pi

#define PI (PCS) M_PI

#ifdef SINGLE
#define EPSILON (float)6e-08
#else
#define EPSILON (double)1.1e-16
#endif

#define BLOCKSIZE 16

#define SPEEDOFLIGHT 299792458.0

#define MAX_CUFFT_ELEM 128e6 // may change for different kind of GPUs

//random11 and rand01
// Random numbers: crappy unif random number generator in [0,1):
//#define rand01() (((PCS)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((PCS)rand() / RAND_MAX)
// unif[-1,1):
#define randm11() (2 * rand01() - (PCS)1.0)

struct conv_opts;
void rescaling_real_invoker(PCS *d_x, PCS scale_ratio, int N);
void rescaling_complex_invoker(CUCPX *d_x, PCS scale_ratio, int N);
void prefix_scan(PCS *d_arr, PCS *d_res, int n, int flag);
// void prefix_scan(int *d_arr, int *d_res, unsigned long int n, int flag);
void get_max_min(PCS &max, PCS &min, PCS *d_array, int n);
int matrix_transpose_invoker(PCS *d_arr, int width, int height);
int matrix_elementwise_multiply_invoker(CUCPX *a, PCS *b, int N);
int matrix_elementwise_divide_invoker(CUCPX *a, PCS *b, int N);
void set_nhg_w(PCS S, PCS X, conv_opts spopts, int &nf, PCS &h, PCS &gam);
void final_hive_plane_bin_mapping(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
                                  int *sortidx_bin, int *histo_count, int *hive_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int *nhive, int pirange);
void part_histogram_3d_sparse_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int plane, int pirange);
void part_mapping_based_gather_3d_invoker(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
                                          int *sortidx_bin, int *histo_count, int2 *se_loc, int N_v, int nf1, int nf2, int nf3, int plane, int init_scan_value, int pirange);
void part_mapping_based_gather_3d_invoker(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
                                          int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int *nhive, int cube_id, int cube_z,
                                          int init_scan_value, int pirange);
void histogram_3d_cube_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count,
                               int N_v, int nf1, int nf2, int nf3, int *hivesize, int *nhive, int cube_id, int cube_z, int pirange);
void histogram_3d_sparse_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int *nhive, int pirange);
void mapping_based_gather_3d_invoker(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
                                     int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int *nhive, int method, int pirange);
void taylor_series_approx_factors(PCS *c0, PCS *c1, PCS *c2, PCS *c3, double beta, int N);
void taylor_series_approx_factors(PCS *c0, double beta, int N, int N_order, int func_type);

void mapping_based_gather_3d_ignore_ibo_invoker(PCS *x, PCS *y, PCS *z, CUCPX *c, PCS *x_out, PCS *y_out, PCS *z_out, CUCPX *c_out,
                                                int *sortidx_bin, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize, int *nhive, int pirange);
void histogram_3d_ignore_inbinorder_invoker(PCS *x, PCS *y, PCS *z, int *sortidx_hive, int *histo_count, int N_v, int nf1, int nf2, int nf3, int *hivesize,
                                            int *nhive, int pirange);
void part_mapping_3d_invoker(PCS *x, PCS *y, PCS *z, CUCPX *c, int *idx_arr, int *sortidx_bin, int *histo_count, int N_v,
                             int nf1, int nf2, int nf3, int *hivesize, int *nhive, int cube_id, int cube_z, int init_scan_value, int pirange);
#endif