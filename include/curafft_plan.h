#ifndef __CURAFFT_PLAN_H__
#define __CURAFFT_PLAN_H__

#include <cstdlib>
#include <cufft.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "datatype.h"
#include "curafft_opts.h"
#include "utils.h"
#include "../contrib/common.h"
#include "../contrib/utils_fp.h"

#define MAX_KERNEL_WIDTH 16

#undef CURAFFT_PLAN

#ifdef SINGLE
#define CURAFFT_PLAN curafftf_plan
#else
#define CURAFFT_PLAN curafft_plan
#endif

struct t3attrib
{
	PCS i_half_width[3];
	PCS o_half_width[3];

	PCS i_center[3];
	PCS o_center[3];

	PCS gamma[3];
	PCS h[3];
};

struct conv_opts
{
	/*
    options for convolutional gridding process
    kw - w, the kernel width (number of grid cells)
    direction - 1 means  NU->U, 0 means  interpolate U->NU 
    pirange - 0: coords in [0,N), 1 coords in [-pi,pi) or can be shifted into pirange, 2 coords is nature number for scaling
    upsampfac - sigma, upsampling factor, default 2.0
    ES_beta
    ES_halfwidth
    ES_c
  */
	int kw; //kernel width // also need to take factors in improved ws into consideration
	int direction;
	int pirange; // pirange or nature number
	PCS upsampfac;
	// ES kernel specific...
	PCS ES_beta;
	PCS ES_halfwidth;
	PCS ES_c; //default 4/kw^2 for reusing
};

struct CURAFFT_PLAN
{
	curafft_opts opts;
	conv_opts copts;
	//cufft
	cufftHandle fftplan;
	cufftHandle fftplan_l; // when elements number is larger than 128million, one more plan is needed.
	//A stream in CUDA is a sequence of operations that execute on the device in the order in which they are issued
	//by the host code. While operations within a stream are guaranteed to execute in the prescribed order, operations
	//in different streams can be interleaved and, when possible, they can even run concurrently.
	cudaStream_t *streams;

	t3attrib ta; //type 3 attributes

	//suppose the N_u = N_l
	PCS *d_u;
	PCS *d_v;
	PCS *d_w;
	CUCPX *d_c;

	// specify for type 3
	PCS *d_x; // out
	PCS *d_y;
	PCS *d_z;

	int dim; //dimension support for 1,2,3D
	int type;
	int mode_flag; // FFTW (0) style or CMCL-compatible mode ordering (1)
	int M;		   //NU
	int nf1;	   // UPTS after upsampling
	int nf2;
	int nf3; //number of w after gridding
	int ms;	 // number of Fourier modes N1
	int mt;	 // N2
	int mu;	 // N3
	int ntransf;
	int iflag;
	int batchsize;
	int execute_flow; //may be useless

	//int totalnumsubprob;
	int byte_now;	 //always be set to be 0
	PCS *fwkerhalf1; //used for not just spread only
	PCS *fwkerhalf2;
	PCS *fwkerhalf3;

	CUCPX *fw; // conv res
	CUCPX *fk; // fft res

	int hivesize[3];
	int *hive_count;
	int *histo_count;
	int *sortidx_bin;
	int2 *se_loc;

	// int *idxnupts;	 //length: #nupts, index of the nupts in the bin-sorted order (size is M) abs location in bin
	// int *sortidx;	 //length: #nupts, order inside the bin the nupt belongs to (size is M) local position in bin

	// //----for GM-sort method----
	// int *binsize;	  //length: #bins, number of nonuniform ponits in each bin //one bin can contain add to gpu_binsizex*gpu_binsizey points
	// int *binstartpts; //length: #bins, exclusive scan of array binsize // binsize after scan

	/*
	// Arrays that used in subprob method
	int *numsubprob; //length: #bins,  number of subproblems in each bin
	
	int *subprob_to_bin;//length: #subproblems, the bin the subproblem works on 
	int *subprobstartpts;//length: #bins, exclusive scan of array numsubprob
    
	// Extra arrays for Paul's method
	int *finegridsize;
	int *fgstartpts;
    
	// Arrays for 3d (need to sort out)
	int *numnupts;
	int *subprob_to_nupts;
    */
};

#endif
