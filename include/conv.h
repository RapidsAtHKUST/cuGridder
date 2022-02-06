#ifndef __CONV_CUH__
#define __CONV_CUH__

#include <stdlib.h>
#include "utils.h"
#include "datatype.h"
#include "curafft_plan.h"

// #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// #else
// __device__ double atomicAdd(double *a, double b) { return b; }
// #endif

#define SHARED_SIZE_3D_OUTPUT_HIVE 512
#define SHARED_SIZE_3D_HIVE 1024
#define SHARED_SIZE_SEG 512
// NU coord handling macro: if p is true, rescales from [-pi,pi) to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
// RESCALE is from Barnett 2/7/17.
#define RESCALE(x, N, p) (p ? ((x * M_1_2PI + (x < -PI ? 1.5 : (x >= PI ? -0.5 : 0.5))) * N) : (x < 0 ? x + N : (x >= N ? x - N : x)))

// p is not ture, shift to [-pi, pi) and then rescale, p is true just rescale.
//need to revise, need to combine rescale
#define SHIFT_RESCALE(x, N, p) ((p ? x : ((x - floor(x / M_2PI) * M_2PI) - ((x - floor(x / M_2PI) * M_2PI) >= PI) * M_2PI)) * M_1_2PI + 0.5) * N

void set_ker_eval_lut(PCS *h_c0, PCS *h_c1, PCS *h_c2, PCS *h_c3);

__global__ void conv_1d_nputsdriven(PCS *x, CUCPX *c, CUCPX *fw, int M,
									const int ns, int nf1, PCS es_c, PCS es_beta, int pirange);

__global__ void conv_2d_nputsdriven(PCS *x, PCS *y, CUCPX *c, CUCPX *fw, int M,
									const int ns, int nf1, int nf2, PCS es_c, PCS es_beta, int pirange);

__global__ void conv_3d_nputsdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int M,
									const int ns, int nf1, int nf2, int nf3, PCS es_c, PCS es_beta, int pirange);
// __global__ void conv_3d_outputdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, int *nhive,
// 		int *hivesize, const int ns, int nf1, int nf2, int nf3, PCS es_c, PCS es_beta, int pirange);

// __global__ void counting_hive(int *hive_count, int *histo_count, unsigned long int M, int hivesize);

__global__ void conv_3d_outputdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange);

__global__ void conv_3d_outputdriven_shared(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange);

__global__ void conv_3d_outputdriven_shared_sparse(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange);

// __global__ void conv_3d_outputdriven_shared_hive_lut(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
// 	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange);
__global__ void conv_3d_outputdriven_shared_hive_lut(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, PCS *c0, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, int pirange);

__global__ void partial_conv_3d_outputdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nf3_total, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange, 
	 int init_shift, int up_shift, int c_shift, int down_shift);

__global__ void partial_conv_3d_outputdriven_shared_hive_lut(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, PCS *c0, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nf3_total, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, int pirange, int init_shift, int up_shift, int c_shift, int down_shift);

__global__ void fisrt_hive_plane_nupt(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int M,
									const int ns, int nf1, int nf2, int nf3, PCS es_c, PCS es_beta, int flag, int pirange);

__global__ void conv_3d_nputsdriven_taylor(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, PCS *c0, int M,
									const int ns, int nf1, int nf2, int nf3, int pirange);

__global__ void conv_3d_outputdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, PCS es_c, PCS es_beta, int pirange);

__global__ void conv_3d_outputdriven_t(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange);

__global__ void conv_3d_outputdriven_t1(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nz, int hivesize_x, int hivesize_y, int hivesize_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange);
#endif