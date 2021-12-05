/*
Invoke conv related kernel (device) function
Functions:
  1. get_num_cells
  2. setup_conv_opts
  3. conv_*d_invoker
  4. curafft_conv
  5. curafft_partial_conv
Issue: Revise for batch
*/

#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <cuComplex.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include "conv_interp_invoker.h"
#include "conv.h"
#include "interp.h"
#include "utils.h"
#include "common_utils.h"

int get_num_cells(PCS ms, conv_opts copts)
// type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms.
{
  /*
    Determain the size of the grid
    ms - number of fourier modes (image size)
    copt - contains convolution related parameters
  */
  int nf = (int)(copts.upsampfac*ms);
  if (nf<2*copts.kw) nf=2*copts.kw; // otherwise spread fails
  if (nf<1e11){                                // otherwise will fail anyway
      nf = next235beven(nf, 1);
  }
  return nf;
}

int setup_conv_opts(conv_opts &opts, PCS eps, PCS upsampfac, int pirange, int direction, int kerevalmeth)
{
  /*
    setup conv related components
    follow the setting in  Yu-hsuan Shih (https://github.com/flatironinstitute/cufinufft)
  */
  // handling errors or warnings
  if (upsampfac != 2.0)
  { // nonstandard sigma
    if (kerevalmeth == 1)
    {
      fprintf(stderr, "setup_conv_opts: nonstandard upsampling factor %.3g with kerevalmeth=1\n", (double)upsampfac);
      return 2;
    }
    if (upsampfac <= 1.0)
    {
      fprintf(stderr, "setup_conv_opts: error, upsampling factor %.3g too small\n", (double)upsampfac);
      return 2;
    }
    // calling routine must abort on above errors, since opts is garbage!
    if (upsampfac > 4.0)
      fprintf(stderr, "setup_conv_opts: warning, upsampfac=%.3g is too large\n", (double)upsampfac);
  }

  opts.direction = direction; 
  opts.pirange = pirange; // in range [-pi,pi) or [0,N)
  opts.upsampfac = upsampfac; // upsamling factor

  
  int ier = 0;
  if (eps < EPSILON)
  {
    fprintf(stderr, "setup_conv_opts: warning, eps (tol) is too small, set eps = %.3g.\n", (double)EPSILON);
    eps = EPSILON;
    ier = 1;
  }

  // kernel width  (kw) and ES kernel beta parameter setting
  int kw = std::ceil(-log10(eps / (PCS)10.0));                  // 1 digit per power of ten
  if (upsampfac != 2.0)                                         // override ns for custom sigma
    kw = std::ceil(-log(eps) / (PI * sqrt(1.0 - 1.0 / upsampfac))); // formula, gamma=1
  kw = max(2, kw);  
  // printf("ns is %d\n",kw);                                         
  if (kw > MAX_KERNEL_WIDTH)
  { // clip to match allocated arrays
    fprintf(stderr, "%s warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; clipping to max %d, better to revise sigma and tol.\n", __func__,
            upsampfac, (double)eps, kw, MAX_KERNEL_WIDTH);
    kw = MAX_KERNEL_WIDTH;
    printf("warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; clipping to max %d, better to revise sigma and tol.\n",
            upsampfac, (double)eps, kw, MAX_KERNEL_WIDTH);
  }
  opts.kw = kw;
  opts.ES_halfwidth = (PCS)kw / 2; // constants to help ker eval (except Horner)
  opts.ES_c = 4.0 / (PCS)(kw * kw);

  PCS betaoverns = 2.30; // gives decent betas for default sigma=2.0
  if (kw == 2)
    betaoverns = 2.20; // some small-width tweaks...
  if (kw == 3)
    betaoverns = 2.26;
  if (kw == 4)
    betaoverns = 2.38;
  if (upsampfac != 2.0)
  {                                                      // again, override beta for custom sigma
    PCS gamma = 0.97;                                    // must match devel/gen_all_horner_C_code.m
    betaoverns = gamma * PI * (1 - 1 / (2 * upsampfac)); // formula based on cutoff
  }
  opts.ES_beta = betaoverns * (PCS)kw; // set the kernel beta parameter
  // printf("the value of beta %.3f\n",opts.ES_beta);
  //fprintf(stderr,"setup_spreader: sigma=%.6f, chose ns=%d beta=%.6f\n",(double)upsampfac,ns,(double)opts.ES_beta); // user hasn't set debug yet
  return ier;
}

int bin_mapping(CURAFFT_PLAN *plan, PCS *uvw){
  // +++++++ method and hivesize /// later support other dim 1,2 +++ dim condition 
  if(plan->dim==3){
    int M = plan->M;
    int nf1 = plan->nf1;
    int nf2 = plan->nf2;
    int nf3 = plan->nf3;
    int method = plan->opts.gpu_gridder_method;
    PCS *d_u_out, *d_v_out, *d_w_out;
    CUCPX *d_c_out;
    checkCudaErrors(cudaMalloc((void **)&d_u_out, sizeof(PCS)*M));
    checkCudaErrors(cudaMalloc((void **)&d_v_out, sizeof(PCS)*M));
    checkCudaErrors(cudaMalloc((void **)&d_w_out,sizeof(PCS)*M));
    checkCudaErrors(cudaMalloc((void **)&d_c_out,sizeof(CUCPX)*M));
    checkCudaErrors(cudaMalloc((void **)&plan->sortidx_bin,sizeof(int)*M));
    if(method==2||method==3||method==4){ // sufficient memory // a litter messy
      int nhive[3];
      nhive[0] = (nf1-1)/plan->hivesize[0] + 1;
      nhive[1] = (nf2-1)/plan->hivesize[1] + 1;
      nhive[2] = (nf3-1)/plan->hivesize[2] + 1;

      if((nhive[0]<3||nhive[1]<3||nhive[2]<3)&&method!=2){
        printf("exit one hive size smaller than 3, automatically switch to method 2\n");
        method = 2;
        plan->opts.gpu_gridder_method = 2;
      }
    
      unsigned long int histo_count_size = nhive[0]*plan->hivesize[0]; // padding
      histo_count_size *= nhive[1]*plan->hivesize[1];
      histo_count_size *= nhive[2]*plan->hivesize[2];
      histo_count_size ++;
      // printf("%lu\n",histo_count_size);
      // printf("nf1 nf2 nf3 %d, %d, %d\n",plan->nf1,plan->nf2,plan->nf3);
      int hive_count_size = nhive[0] * nhive[1] * nhive[2] + 1;
      // printf("%d\n",hive_count_size);
      
      // show_mem_usage();
      checkCudaErrors(cudaMalloc((void **)&plan->histo_count,sizeof(int)*(histo_count_size)));
      // show_mem_usage();

      checkCudaErrors(cudaMalloc((void **)&plan->hive_count,sizeof(int)*(hive_count_size)));
      checkCudaErrors(cudaMemset(plan->histo_count,0,sizeof(int)*(histo_count_size)));
      checkCudaErrors(cudaDeviceSynchronize());
      histogram_3d_sparse_invoker(plan->d_u,plan->d_v,plan->d_w,plan->sortidx_bin,plan->histo_count,M,nf1,nf2,nf3,plan->hivesize,nhive,plan->copts.pirange);
      // show_mem_usage();
      prefix_scan(plan->histo_count,plan->histo_count,histo_count_size,0);
      // calculate hive count
      // printf("%d\n",plan->hivesize[0]*plan->hivesize[1]*plan->hivesize[2]);
      counting_hive_invoker(plan->hive_count, plan->histo_count, hive_count_size, plan->hivesize[0]*plan->hivesize[1]*plan->hivesize[2]);
      
      mapping_based_gather_3d_invoker(plan->d_u,plan->d_v,plan->d_w,plan->d_c,d_u_out,d_v_out,d_w_out,d_c_out,plan->sortidx_bin,plan->histo_count,M,nf1,nf2,nf3,plan->hivesize,nhive,method,plan->copts.pirange);
    }
    else if(method==1){ // partical mapping
      checkCudaErrors(cudaMalloc((void **)&plan->histo_count,sizeof(int)*(nf1*nf2+1)));
      checkCudaErrors(cudaMalloc((void **)&plan->se_loc,sizeof(int2)*M));
      checkCudaErrors(cudaMemset(plan->histo_count,0,sizeof(int)*(nf1*nf2+1)));
      int init_scan_value = 0;
      for(int i=0; i<nf3; i++){
        part_histogram_3d_sparse_invoker(plan->d_u,plan->d_v,plan->d_w,plan->sortidx_bin,plan->histo_count,M,nf1,nf2,nf3,i,plan->copts.pirange);
        prefix_scan(plan->histo_count,plan->histo_count,nf1*nf2+1,0);
        part_mapping_based_gather_3d_invoker(plan->d_u,plan->d_v,plan->d_w,plan->d_c,d_u_out,d_v_out,d_w_out,d_c_out,plan->sortidx_bin,plan->histo_count,plan->se_loc,M,nf1,nf2,nf3,i,init_scan_value,plan->copts.pirange);
        int last_value;
        checkCudaErrors(cudaMemcpy(&last_value,plan->histo_count+nf1*nf2,sizeof(int),cudaMemcpyDeviceToHost));
        init_scan_value += last_value;
        /// ++++ plane array
        checkCudaErrors(cudaMemset(plan->histo_count,0,sizeof(int)*(nf1*nf2+1)));
      }
      if(method == 1)checkCudaErrors(cudaFree(plan->se_loc));
    }
    if(uvw==NULL){
      checkCudaErrors(cudaFree(plan->d_u));
      checkCudaErrors(cudaFree(plan->d_v));
      checkCudaErrors(cudaFree(plan->d_w));
      checkCudaErrors(cudaFree(plan->d_c));
    }
    else{
      checkCudaErrors(cudaFree(uvw));
    }
    checkCudaErrors(cudaFree(plan->sortidx_bin));
    checkCudaErrors(cudaFree(plan->histo_count));
    plan->d_u = d_u_out;
    plan->d_v = d_v_out;
    plan->d_w = d_w_out;
    plan->d_c = d_c_out;
  }
  return 0;
}

void final_hive_plane(CURAFFT_PLAN *plan, unsigned long int histo_count_size){
  int nhive[3];
  nhive[0] = (plan->nf1-1)/plan->hivesize[0] + 1;
  nhive[1] = (plan->nf2-1)/plan->hivesize[1] + 1;
  nhive[2] = (plan->nf3-1)/plan->hivesize[2] + 1;
  // set histo 0
  checkCudaErrors(cudaMemset(plan->histo_count,0,histo_count_size*sizeof(int)));
  // plan->hive_count+zuihou
  int *hive_count = plan->hive_count + nhive[0] * nhive[1] * nhive[2] * 2 + 1 - nhive[0]*nhive[1];
  final_hive_plane_bin_mapping(plan->d_u, plan->d_v, plan->d_w, plan->d_c, plan->d_u_out, plan->d_v_out, plan->d_w_out, plan->d_c_out,
    plan->sortidx_bin, plan->histo_count, hive_count, plan->M, plan->nf1, plan->nf2, plan->mem_limit, plan->hivesize, nhive, plan->copts.pirange);
  checkCudaErrors(cudaMemset(plan->histo_count,0,histo_count_size*sizeof(int)));
}

int part_bin_mapping(CURAFFT_PLAN *plan, PCS *d_u_out, PCS *d_v_out, PCS *d_w_out, CUCPX *d_c_out, unsigned long int histo_count_size, int cube_id, int &initial){
  int M = plan->M;
  int nf1 = plan->nf1;
  int nf2 = plan->nf2;
  int nf3 = plan->nf3;
  // int method = plan->opts.gpu_gridder_method;
  checkCudaErrors(cudaMalloc((void **)&plan->sortidx_bin,sizeof(int)*M));
  // if(plan->nf3<24){
  //   PCS *d_u_out, *d_v_out, *d_w_out;
  //   CUCPX *d_c_out;
  //   // one plane by one plane
  //   checkCudaErrors(cudaMalloc((void **)&d_u_out, sizeof(PCS)*M));
  //   checkCudaErrors(cudaMalloc((void **)&d_v_out, sizeof(PCS)*M));
  //   checkCudaErrors(cudaMalloc((void **)&d_w_out,sizeof(PCS)*M));
  //   checkCudaErrors(cudaMalloc((void **)&d_c_out,sizeof(CUCPX)*M));
  //   checkCudaErrors(cudaMalloc((void **)&plan->histo_count,sizeof(int)*(nf1*nf2+1)));
  //   checkCudaErrors(cudaMalloc((void **)&plan->se_loc,sizeof(int2)*M));
  //   checkCudaErrors(cudaMemset(plan->histo_count,0,sizeof(int)*(nf1*nf2+1)));
  //   int init_scan_value = 0;
  //   for(int i=0; i<nf3; i++){
  //     part_histogram_3d_sparse_invoker(plan->d_u,plan->d_v,plan->d_w,plan->sortidx_bin,plan->histo_count,M,nf1,nf2,nf3,i,plan->copts.pirange);
  //     prefix_scan(plan->histo_count,plan->histo_count,nf1*nf2+1,0);
  //     part_mapping_based_gather_3d_invoker(plan->d_u,plan->d_v,plan->d_w,plan->d_c,d_u_out,d_v_out,d_w_out,d_c_out,plan->sortidx_bin,plan->histo_count,plan->se_loc,M,nf1,nf2,nf3,i,init_scan_value,plan->copts.pirange);
  //     int last_value;
  //     checkCudaErrors(cudaMemcpy(&last_value,plan->histo_count+nf1*nf2,sizeof(int),cudaMemcpyDeviceToHost));
  //     init_scan_value += last_value;
  //     /// ++++ plane array
  //     checkCudaErrors(cudaMemset(plan->histo_count,0,sizeof(int)*(nf1*nf2+1)));
  //   }
  //   if(method == 1)checkCudaErrors(cudaFree(plan->se_loc));
  // }
  // else{
    int nhive[3];
    nhive[0] = (nf1-1)/plan->hivesize[0] + 1;
    nhive[1] = (nf2-1)/plan->hivesize[1] + 1;
    nhive[2] = (nf3-1)/plan->hivesize[2] + 1;

    // consider cube_id = 0 as special case

    int *hive_count;
    int hive_count_size = nhive[0] * nhive[1] * nhive[2] + 1;
    if(cube_id % 2){
      hive_count = plan->hive_count + hive_count_size; // odd 
    }
    else{
      hive_count = plan->hive_count;
    }
    checkCudaErrors(cudaMemcpy(plan->hive_count+2*hive_count_size, hive_count+hive_count_size-nhive[0]*nhive[1]-1,sizeof(int)*(nhive[0]*nhive[1]+1),cudaMemcpyDeviceToDevice));
    
    histogram_3d_cube_invoker(plan->d_u,plan->d_v,plan->d_w,plan->sortidx_bin,plan->histo_count,M,nf1,nf2,plan->mem_limit,plan->hivesize,nhive,cube_id,nf3,plan->copts.pirange);
    prefix_scan(plan->histo_count,plan->histo_count,histo_count_size,0);
    int last_value;
    checkCudaErrors(cudaMemcpy(&last_value,plan->histo_count+histo_count_size-1,sizeof(int),cudaMemcpyDeviceToHost));
    counting_hive_invoker(hive_count, plan->histo_count, hive_count_size, plan->hivesize[0]*plan->hivesize[1]*plan->hivesize[2],initial);
    // if(cube_id==3){
    //   int *h_hive_count = (int *) malloc(sizeof(int)*hive_count_size);
    //   checkCudaErrors(cudaMemcpy(h_hive_count,plan->hive_count,sizeof(int)*hive_count_size,cudaMemcpyDeviceToHost));
    //   for(int ii=0; ii<1024;ii++){
    //     printf("%d ",h_hive_count[ii]);
    //   }
    //   printf("\n");
    // }
    part_mapping_based_gather_3d_invoker(plan->d_u,plan->d_v,plan->d_w,plan->d_c,d_u_out,d_v_out,d_w_out,d_c_out,plan->sortidx_bin,plan->histo_count,M,nf1,nf2,plan->mem_limit,plan->hivesize,nhive,cube_id,nf3,initial,plan->copts.pirange);
    initial += last_value;
  // }
  // checkCudaErrors(cudaFree(plan->sortidx_bin));
  // checkCudaErrors(cudaFree(plan->histo_count));
  return 0;
}

int part_bin_mapping_pre(CURAFFT_PLAN *plan, int *temp_station, int &initial){
  // memory malloc
  int nhive[3];
  nhive[0] = (plan->nf1-1)/plan->hivesize[0] + 1;
  nhive[1] = (plan->nf2-1)/plan->hivesize[1] + 1;
  nhive[2] = (plan->nf3-1)/plan->hivesize[2] + 1;
  int M = plan->M;
  int hive_count_size = nhive[0] * nhive[1] * nhive[2] * 2 + nhive[0] * nhive[1] + 3;
  checkCudaErrors(cudaMalloc((void **)&plan->hive_count,sizeof(int)*(hive_count_size))); // malloc once
  
  checkCudaErrors(cudaMalloc((void **)&plan->d_u_out, sizeof(PCS)*M));
  checkCudaErrors(cudaMalloc((void **)&plan->d_v_out, sizeof(PCS)*M));
  checkCudaErrors(cudaMalloc((void **)&plan->d_w_out,sizeof(PCS)*M));
  checkCudaErrors(cudaMalloc((void **)&plan->d_c_out,sizeof(CUCPX)*M));

  unsigned long int histo_count_size = nhive[0]*plan->hivesize[0]; // padding
  histo_count_size *= nhive[1]*plan->hivesize[1];
  histo_count_size *= nhive[2]*plan->hivesize[2];
  histo_count_size ++;
  printf("nf1 nf2 nf3 %d,%d,%d\n",plan->nf1,plan->nf2,plan->nf3);
  show_mem_usage();
  checkCudaErrors(cudaMalloc((void **)&plan->histo_count,sizeof(int)*(histo_count_size)));
  checkCudaErrors(cudaMemset(plan->histo_count,0,sizeof(int)*(histo_count_size)));
  plan->initial = 0;
  part_bin_mapping(plan, plan->d_u_out, plan->d_v_out, plan->d_w_out, plan->d_c_out, histo_count_size, 0, plan->initial); // first cube
  // calcalute the final hive plane
  final_hive_plane(plan, histo_count_size);

  // save the first hive plane for final hive
  checkCudaErrors(cudaMemcpy(temp_station,plan->hive_count,sizeof(int)*nhive[0]*nhive[1]+1,cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(plan->histo_count));
  checkCudaErrors(cudaFree(plan->sortidx_bin));

  // set histro 0 and repeat until the last cube
  return 0;
}

int conv_1d_invoker(int nf1, int M, CURAFFT_PLAN *plan){
  /*
    convolution invoker, invoke the kernel function
    nf1 - grid size in 1D
    M - number of points
  */
  dim3 grid;
  dim3 block;
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256; // 256 threads per block
    grid.x = (M - 1) / block.x + 1; // number of blocks

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    conv_1d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, plan->copts.ES_c, plan->copts.ES_beta, 
                                          plan->copts.pirange);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  return 0;
}

int conv_2d_invoker(int nf1, int nf2, int M, CURAFFT_PLAN *plan)
{

  dim3 grid;
  dim3 block;
  
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    conv_2d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, plan->copts.ES_c, plan->copts.ES_beta, 
                                          plan->copts.pirange);
    

    checkCudaErrors(cudaDeviceSynchronize());
  }

  return 0;
}

int conv_3d_invoker(int nf1, int nf2, int nf3, int M, CURAFFT_PLAN *plan)
{
  // cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
  dim3 grid;
  dim3 block;
  int method = plan->opts.gpu_gridder_method;
  // cudaEventRecord(start);
  if (method == 0 || method==1)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;
    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    conv_3d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_w, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, nf3, plan->copts.ES_c, plan->copts.ES_beta,
                                          plan->copts.pirange);
    checkCudaErrors(cudaDeviceSynchronize());
    // float milliseconds = 0;
		// cudaEventRecord(stop);
		// cudaEventSynchronize(stop);
		// cudaEventElapsedTime(&milliseconds, start, stop);
		// printf("[time  ] \tKernel Spread_3d_NUptsdriven (%d)\t%.3g ms\n", 
		// milliseconds,plan->opts.gpu_kerevalmeth);
    
  }
  else{
    block.x = plan->hivesize[0]*plan->hivesize[1]*plan->hivesize[2];
    int nhive[3];
    nhive[0] = (nf1-1)/plan->hivesize[0] + 1;
    nhive[1] = (nf2-1)/plan->hivesize[1] + 1;
    nhive[2] = (nf3-1)/plan->hivesize[2] + 1;
    grid.x = nhive[0] * nhive[1] * nhive[2];
    // printf("blocksize %d, grid %d %d %d\n", block.x, nhive[0], nhive[1], nhive[2]);
    if(method==2)
    conv_3d_outputdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_w, plan->d_c, plan->fw, plan->hive_count, plan->copts.kw, 
                                          nf1, nf2, nf3, plan->hivesize[0]*nhive[0], plan->hivesize[1]*nhive[1], plan->hivesize[2]*nhive[2], 
                                          nhive[0], nhive[1], nhive[2], plan->copts.ES_c, plan->copts.ES_beta, plan->copts.pirange);
    if(method==3){
      conv_3d_outputdriven_shared_sparse<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_w, plan->d_c, plan->fw, plan->hive_count, plan->copts.kw, 
                                          nf1, nf2, nf3, plan->hivesize[0]*nhive[0], plan->hivesize[1]*nhive[1], plan->hivesize[2]*nhive[2], 
                                          nhive[0], nhive[1], nhive[2], plan->copts.ES_c, plan->copts.ES_beta, plan->copts.pirange);
      // free hive count
    }

    if(method==4){
      // PCS *h_lut = (PCS *)malloc(sizeof(PCS)*LOOKUP_TABLE_SIZE);
      // memset(h_lut,0,sizeof(PCS)*LOOKUP_TABLE_SIZE);
      // set_ker_eval_lut(h_lut);
      conv_3d_outputdriven_shared_hive_lut<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_w, plan->d_c, plan->fw, plan->c0, plan->hive_count, plan->copts.kw, 
                                          nf1, nf2, nf3, plan->hivesize[0]*nhive[0], plan->hivesize[1]*nhive[1], plan->hivesize[2]*nhive[2], 
                                          nhive[0], nhive[1], nhive[2], plan->copts.pirange);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(plan->hive_count));
    if(plan->opts.gpu_kerevalmeth)checkCudaErrors(cudaFree(plan->c0));
    
    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        exit(1);
        // Possibly: exit(-1) if program cannot continue....
    }
    // float milliseconds = 0;
		// cudaEventRecord(stop);
		// cudaEventSynchronize(stop);
		// cudaEventElapsedTime(&milliseconds, start, stop);
		// printf("[time  ] \tKernel Spread_3d_NUptsdriven (%d)\t%.3g ms\n", 
		// milliseconds,plan->opts.gpu_kerevalmeth);
    // checkCudaErrors(cudaFree(plan->hive_count));
    
  }

  return 0;
}

int curafft_conv(CURAFFT_PLAN * plan)
{
  /*
  ---- convolution opertion ----
  */

  int ier = 0;
  int nf1 = plan->nf1;
  int nf2 = plan->nf2;
  int nf3 = plan->nf3;
  int M = plan->M;
  // printf("w_term_method %d\n",plan->w_term_method);
  switch (plan->dim)
  {
    case 1:{
      conv_1d_invoker(nf1, M, plan);
      break;
    }
    case 2:{
      conv_2d_invoker(nf1, nf2, M, plan);
      break;
    }
    case 3:{
      // if(plan->opts.gpu_sort){
      //   bin_mapping(plan);
      // }
      // show_mem_usage();
      // printf("size %d\n", plan->nf1*plan->nf2*plan->nf3);
      if(plan->fw==NULL){
        unsigned long long int fw_size = plan->nf1;
        fw_size *= plan->nf2;
        fw_size *= plan->nf3;
        checkCudaErrors(cudaMalloc(&plan->fw, fw_size * sizeof(CUCPX)));
        checkCudaErrors(cudaMemset(plan->fw,0, fw_size * sizeof(CUCPX)));
      }
      // show_mem_usage();
      conv_3d_invoker(nf1, nf2, nf3, M, plan);
      break;
    }
    default:
      ier = 1; // error
      break;
  }

  return ier;
}


int interp_1d_invoker(int nf1, int M, CURAFFT_PLAN *plan){
  /*
    convolution invoker, invoke the kernel function
    nf1 - grid size in 1D
    M - number of points
  */
  dim3 grid;
  dim3 block;
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256; // 256 threads per block
    grid.x = (M - 1) / block.x + 1; // number of blocks

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    interp_1d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, plan->copts.ES_c, plan->copts.ES_beta, 
                                          plan->copts.pirange);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  return 0;
}

int interp_2d_invoker(int nf1, int nf2, int M, CURAFFT_PLAN *plan)
{

  dim3 grid;
  dim3 block;
  
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    interp_2d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, plan->copts.ES_c, plan->copts.ES_beta, 
                                          plan->copts.pirange);
    

    checkCudaErrors(cudaDeviceSynchronize());
  }

  return 0;
}

int interp_3d_invoker(int nf1, int nf2, int nf3, int M, CURAFFT_PLAN *plan)
{

  dim3 grid;
  dim3 block;
  
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;
    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    interp_3d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_w, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, nf3, plan->copts.ES_c, plan->copts.ES_beta,
                                          plan->copts.pirange);
    

    checkCudaErrors(cudaDeviceSynchronize());
  }

  return 0;
}

int curafft_interp(CURAFFT_PLAN * plan)
{
  /*
  ---- convolution opertion ----
  */

  int ier = 0;
  int nf1 = plan->nf1;
  int nf2 = plan->nf2;
  int nf3 = plan->nf3;
  int M = plan->M;
  // printf("w_term_method %d\n",plan->w_term_method);
  switch (plan->dim)
  {
  case 1:
    interp_1d_invoker(nf1, M, plan);
    break;
  case 2:
    interp_2d_invoker(nf1, nf2, M, plan);
    break;
  case 3:
    interp_3d_invoker(nf1, nf2, nf3, M, plan);
    break;
  default:
    ier = 1; // error
    break;
  }

  return ier;
}

int partial_conv_3d_invoker(int nf1, int nf2, int nf3, int M, int init_shift, int up_shift, int c_shift, int down_shift, CURAFFT_PLAN *plan){
  dim3 grid;
  dim3 block;
  int method = plan->opts.gpu_gridder_method;
  block.x = plan->hivesize[0]*plan->hivesize[1]*plan->hivesize[2];
  int nhive[3];
  nhive[0] = (nf1-1)/plan->hivesize[0] + 1;
  nhive[1] = (nf2-1)/plan->hivesize[1] + 1;
  nhive[2] = (nf3-1)/plan->hivesize[2] + 1;
  grid.x = nhive[0] * nhive[1] * nhive[2];
  // printf("blocksize %d, grid %d %d %d\n", block.x, nhive[0], nhive[1], nhive[2]);
  if(method==2)
  partial_conv_3d_outputdriven<<<grid, block>>>(plan->d_u_out, plan->d_v_out, plan->d_w_out, plan->d_c_out, plan->fw, plan->hive_count, plan->copts.kw, 
                                        nf1, nf2, nf3, plan->mem_limit, plan->hivesize[0]*nhive[0], plan->hivesize[1]*nhive[1], plan->hivesize[2]*nhive[2], 
                                        nhive[0], nhive[1], nhive[2], plan->copts.ES_c, plan->copts.ES_beta, plan->copts.pirange, init_shift, up_shift, c_shift, down_shift);

  else if(method==4){
    // PCS *h_lut = (PCS *)malloc(sizeof(PCS)*LOOKUP_TABLE_SIZE);
    // memset(h_lut,0,sizeof(PCS)*LOOKUP_TABLE_SIZE);
    // set_ker_eval_lut(h_lut);
    partial_conv_3d_outputdriven_shared_hive_lut<<<grid, block>>>(plan->d_u_out, plan->d_v_out, plan->d_w_out, plan->d_c_out, plan->fw, plan->c0, plan->hive_count, plan->copts.kw, 
                                        nf1, nf2, nf3, plan->mem_limit, plan->hivesize[0]*nhive[0], plan->hivesize[1]*nhive[1], plan->hivesize[2]*nhive[2], 
                                        nhive[0], nhive[1], nhive[2], plan->copts.pirange, init_shift, up_shift, c_shift, down_shift);
  }
  else {printf("please select other methods\n"); return 1;}
  checkCudaErrors(cudaDeviceSynchronize());
  // checkCudaErrors(cudaFree(plan->hive_count));
  // if(plan->opts.gpu_kerevalmeth)checkCudaErrors(cudaFree(plan->c0));
  
  cudaError_t err = cudaGetLastError();

  if ( err != cudaSuccess )
  {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));       
      exit(1);
      // Possibly: exit(-1) if program cannot continue....
  }
  return 0;
}


int curaff_partial_conv(CURAFFT_PLAN *plan, int init_shift, int up_shift, int c_shift, int down_shift){
  
  int ier = 0;
  int nf1 = plan->nf1;
  int nf2 = plan->nf2;
  int nf3 = plan->nf3;
  int M = plan->M;
  if(plan->fw==NULL){
    printf("error, not assign mem for fw\n");
  }

  ier = partial_conv_3d_invoker(nf1, nf2, nf3, M, init_shift, up_shift, c_shift, down_shift, plan);

  return ier;
}