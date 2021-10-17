#ifndef COMMON_H
#define COMMON_H

#include "datatype.h"
#include "utils_fp.h"
#include "utils.h"

// revise
// #include "spreadinterp.h"

// constants needed within common
#define MAX_NQUAD 100 // max number of positive quadr nodes
// increase this if you need >1TB RAM...
#define MAX_NF (int)1e11 // too big to ever succeed (next235 takes 1s)

//#define MAX(a,b) (a>b) ? a : b  // but we use std::max instead
#define MIN(a, b) (a < b) ? a : b

// allow compile-time switch off of openmp, so compilation without any openmp
// is done (Note: _OPENMP is automatically set by -fopenmp compile flag)
#ifdef _OPENMP
#include <omp.h>
// point to actual omp utils
#define MY_OMP_GET_NUM_THREADS() omp_get_num_threads()
#define MY_OMP_GET_MAX_THREADS() omp_get_max_threads()
#define MY_OMP_GET_THREAD_NUM() omp_get_thread_num()
#define MY_OMP_SET_NUM_THREADS(x) omp_set_num_threads(x)
#define MY_OMP_SET_NESTED(x) omp_set_nested(x)
#else
// non-omp safe dummy versions of omp utils
#define MY_OMP_GET_NUM_THREADS() 1
#define MY_OMP_GET_MAX_THREADS() 1
#define MY_OMP_GET_THREAD_NUM() 0
#define MY_OMP_SET_NUM_THREADS(x)
#define MY_OMP_SET_NESTED(x)
#endif

struct conv_opts;

// common.cpp provides...
void set_nhg_type3(PCS S, PCS X, conv_opts spopts, int &nf, PCS &h, PCS &gam);
void onedim_fseries_kernel_seq(int nf, PCS *fwkerhalf, conv_opts opts);
void onedim_fseries_kernel(int nf, PCS *fwkerhalf, conv_opts opts);
#endif // COMMON_H
