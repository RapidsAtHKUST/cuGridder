#ifndef __INTERP_CUH__
#define __INTERP_CUH__

#include <stdlib.h>
#include "utils.h"
#include "datatype.h"
#include "curafft_plan.h"

// NU coord handling macro: if p is true, rescales from [-pi,pi) to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
// RESCALE is from Barnett 2/7/17.
#define RESCALE(x, N, p) (p ? ((x * M_1_2PI + (x < -PI ? 1.5 : (x >= PI ? -0.5 : 0.5))) * N) : (x < 0 ? x + N : (x >= N ? x - N : x)))

// p is not ture, shift to [-pi, pi) and then rescale, p is true just rescale.
//need to revise, need to combine rescale
#define SHIFT_RESCALE(x, N, p) ((p ? x : ((x - floor(x / M_2PI) * M_2PI) - ((x - floor(x / M_2PI) * M_2PI) >= PI) * M_2PI)) * M_1_2PI + 0.5) * N

__global__ void interp_1d_nputsdriven(PCS *x, CUCPX *c, CUCPX *fw, int M,
									  const int ns, int nf1, PCS es_c, PCS es_beta, int pirange);

__global__ void interp_2d_nputsdriven(PCS *x, PCS *y, CUCPX *c, CUCPX *fw, int M,
									  const int ns, int nf1, int nf2, PCS es_c, PCS es_beta, int pirange);

__global__ void interp_3d_nputsdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int M,
									  const int ns, int nf1, int nf2, int nf3, PCS es_c, PCS es_beta, int pirange);

#endif