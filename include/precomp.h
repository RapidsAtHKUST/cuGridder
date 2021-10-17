#ifndef __PRECOMP__H__
#define __PRECOMP__H__

#include <cuda.h>
#include <helper_cuda.h>
#include "datatype.h"
#include "ragridder_plan.h"
#include "curafft_plan.h"

int explicit_gridder_invoker(ragridder_plan *gridder_plan, PCS e);
void pre_setting(PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_vis, CURAFFT_PLAN *plan, ragridder_plan *gridder_plan);
int w_term_k_generation(PCS *k, int nf1, int nf2, PCS xpixelsize, PCS ypixelsize);
void get_effective_coordinate_invoker(PCS *d_u, PCS *d_v, PCS *d_w, PCS f_over_c, int pirange, int nrow);
void get_effective_coordinate_invoker(PCS *d_u, PCS *d_v, PCS *d_w, PCS f_over_c, int pirange, int nrow, int sign);
#endif