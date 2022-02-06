/*
1. nufft plan setting
2. 1D - dft for w term
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
#include "utils.h"
#include "cuft.h"
#include "conv_interp_invoker.h"
#include "deconv.h"
#include "conv.h"

int cufft_plan_setting(CURAFFT_PLAN *plan){
    // cufft plan setting
    // cufftHandle fftplan;
    int n[] = {plan->nf2, plan->nf1};
    int inembed[] = {plan->nf2, plan->nf1};
    int onembed[] = {plan->nf2, plan->nf1};
    int batchsize = plan->nf3;
    int remain_batch;
    if(MAX_CUFFT_ELEM/plan->nf1/plan->nf2<plan->nf3){
        batchsize = MAX_CUFFT_ELEM/plan->nf1/plan->nf2;
        remain_batch = plan->nf3%batchsize;
        if(remain_batch!=0){
            cufftPlanMany(&plan->fftplan_l, 2, n, inembed, 1, inembed[0] * inembed[1],
                    onembed, 1, onembed[0] * onembed[1], CUFFT_TYPE, remain_batch);
        }
    }
    cufftPlanMany(&plan->fftplan, 2, n, inembed, 1, inembed[0] * inembed[1],
                  onembed, 1, onembed[0] * onembed[1], CUFFT_TYPE, batchsize); //There's a hard limit of roughly 2^27 elements in a plan!!!!!!!!!
    cudaError_t err = cudaGetLastError();
    plan->batchsize = batchsize;
    return remain_batch;
}

__global__ void pre_stage_1(PCS o_center_0, PCS o_center_1, PCS o_center_2, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, int M, int flag)
{
    /*
    prestage 1: cj to cj' (e^(i*x_center*u)*cj) caused by shifting
    Input:
        o_center_*: center of u (v,w) coordinate
        d_u - coordinates
        d_c - value
        M - number of input coordinates 
        flag - -1 or 1
    */
    int idx;
    CUCPX temp;
    for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x)
    {
        PCS phase = o_center_0 * d_u[idx] + (d_v == NULL ? 0 : (o_center_1 * d_v[idx] + (d_w == NULL ? 0 : o_center_2 * d_w[idx])));
        temp.x = d_c[idx].x * cos(phase * flag) - d_c[idx].y * sin(phase * flag);
        temp.y = d_c[idx].x * sin(phase * flag) + d_c[idx].y * cos(phase * flag);
        d_c[idx] = temp;
    }
}

void pre_stage_1_invoker(PCS *o_center, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, int M, int flag){
    int blocksize = 512;
    if(d_c!=NULL){
        if (o_center[0] != 0 || o_center[1] != 0 || o_center[2] != 0)
        {
            // cj to cj'
            pre_stage_1<<<(M - 1) / blocksize + 1, blocksize>>>(o_center[0], o_center[1], o_center[2], d_u, d_v, d_w, d_c, M, flag);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
}

__global__ void pre_stage_2(PCS i_center, PCS o_center, PCS gamma, PCS h, PCS *d_u, PCS *d_x, int M, int N)
{
    /*
    shift and scaling the input coordinates
    Input:
        icenter - center of input
        o_center - center of output
        gamma - scaling ratio
        h - grid unit length
        M - number of u
        N - number of v
    */
    int idx;
    
    for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x)
    {
        d_u[idx] = (d_u[idx] - i_center) / gamma;
    }
    for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        d_x[idx] = (d_x[idx] - o_center) * gamma * h;
    }
}

void pre_stage_2_invoker(PCS *i_center, PCS *o_center, PCS *gamma, PCS *h, PCS *d_u, PCS *d_v, PCS *d_w, PCS *d_x, PCS *d_y, PCS *d_z, CUCPX *d_c, int M, int N1, int N2, int N3){
    int blocksize = 512;
    // uj to uj'', xj to xj'
    pre_stage_2<<<(max(M, N1) - 1) / blocksize + 1, blocksize>>>(i_center[0], o_center[0], gamma[0], h[0], d_u, d_x, M, N1);
    checkCudaErrors(cudaDeviceSynchronize());
    if (d_v != NULL)
    {
        pre_stage_2<<<(max(M, N2) - 1) / blocksize + 1, blocksize>>>(i_center[1], o_center[1], gamma[1], h[1], d_u, d_x, M, N2);
        checkCudaErrors(cudaDeviceSynchronize());
        if (d_w != NULL)
        {
            pre_stage_2<<<(max(M, N3) - 1) / blocksize + 1, blocksize>>>(i_center[2], o_center[2], gamma[2], h[2], d_u, d_x, M, N3);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
}

void pre_stage_invoker(PCS *i_center, PCS *o_center, PCS *gamma, PCS *h, PCS *d_u, PCS *d_v, PCS *d_w, PCS *d_x, PCS *d_y, PCS *d_z, CUCPX *d_c, int M, int N1, int N2, int N3, int flag)
{   
    /*
    Invoke preparing stages transform cj to cj', uj to uj' and xj to xj'
    */
    // Specified for input and output transform
    int blocksize = 512;
    if (flag==1){
        if (o_center[0] != 0 || o_center[1] != 0 || o_center[2] != 0)
        {
            // cj to cj'
            pre_stage_1<<<(M - 1) / blocksize + 1, blocksize>>>(o_center[0], o_center[1], o_center[2], d_u, d_v, d_w, d_c, M, flag);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
    
    
    // uj to uj'', xj to xj'
    pre_stage_2<<<(max(M, N1) - 1) / blocksize + 1, blocksize>>>(i_center[0], o_center[0], gamma[0], h[0], d_u, d_x, M, N1);
    checkCudaErrors(cudaDeviceSynchronize());
    if (d_v != NULL)
    {
        pre_stage_2<<<(max(M, N2) - 1) / blocksize + 1, blocksize>>>(i_center[1], o_center[1], gamma[1], h[1], d_u, d_x, M, N2);
        CHECK(cudaDeviceSynchronize());
        if (d_w != NULL)
        {
            pre_stage_2<<<(max(M, N3) - 1) / blocksize + 1, blocksize>>>(i_center[2], o_center[2], gamma[2], h[2], d_u, d_x, M, N3);
            CHECK(cudaDeviceSynchronize());
        }
    }
}

int cura_prestage(CURAFFT_PLAN *plan){
    if(plan->type!=3){
        int nf1 = plan->nf1;
        int nf2 = 1;
        int nf3 = 1;

        // fourier series
        fourier_series_appro_invoker(plan->fwkerhalf1, plan->copts, nf1/2+1, plan->opts.gpu_kerevalmeth);
        if(plan->dim>1){
            nf2 = plan->nf2;
            fourier_series_appro_invoker(plan->fwkerhalf2, plan->copts, nf2/2+1, plan->opts.gpu_kerevalmeth);
        }
        if(plan->dim>2){
            nf3 = plan->nf3;
            // printf("I am inn\n");
            fourier_series_appro_invoker(plan->fwkerhalf3, plan->copts, nf3/2+1, plan->opts.gpu_kerevalmeth);
        }
        // binmapping
        if(plan->opts.gpu_gridder_method!=0)bin_mapping(plan,NULL); //currently just support 3d

        // fw malloc
        unsigned long long int fw_size = plan->nf1;
        fw_size *= plan->nf2;
        fw_size *= plan->nf3;
        checkCudaErrors(cudaMalloc((void**)&plan->fw, fw_size * sizeof(CUCPX)));
        checkCudaErrors(cudaMemset(plan->fw, 0, fw_size * sizeof(CUCPX)));
    }
    return 0;
}

// cufft_exec
int cura_cufft(CURAFFT_PLAN *plan){
    int batchsize = plan->batchsize;
    int direction = plan->iflag;
    int remain_batch = plan->nf3 % batchsize;
    int elem_num_wb = batchsize*plan->nf1*plan->nf2; // element number whole batches
    int i;
    for(i=0; i<plan->nf3/batchsize; i++){
        CUFFT_EXEC(plan->fftplan, plan->fw+elem_num_wb*i, plan->fw+elem_num_wb*i, direction); // sychronized or not
        checkCudaErrors(cudaDeviceSynchronize());
    }
    if(remain_batch!=0){
        CUFFT_EXEC(plan->fftplan_l, plan->fw+elem_num_wb*i, plan->fw+elem_num_wb*i, direction); // sychronized or not
        checkCudaErrors(cudaDeviceSynchronize());
        if(!plan->mem_limit)cufftDestroy(plan->fftplan_l); // destroy here to save memory
    }
    return 0;
}

int setup_plan(int nf1, int nf2, int nf3, int M, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, CURAFFT_PLAN *plan)
{
    /* different dim will have different setting
    ----plan setting, and related memory allocation----
        nf1, nf2, nf3 - number of UPTS after upsampling
        M - number of NUPTS (num of vis)
        d_u, d_v, d_w - locations
        d_c - value
  */
    int ier = 0;
    plan->d_u = d_u;
    plan->d_v = d_v;
    plan->d_w = d_w;
    plan->d_c = d_c;
    plan->mode_flag = 1; //CMCL mode
    //int upsampfac = plan->copts.upsampfac;

    plan->nf1 = nf1;
    plan->nf2 = nf2;
    plan->nf3 = nf3;

    plan->M = M;

    //plan->maxbatchsize = 1;

    plan->byte_now = 0;
    // No extra memory is needed in nuptsdriven method (case 0, sort 0)
    // switch (plan->opts.gpu_gridder_method)
    // {
    //     case 0:
    //     {
    //         break;
    //     }
    //     case 1:
    //     {
    //         //sorted
    //         // checkCudaErrors(cudaMalloc((void **)&plan->sortidx_bin, sizeof(int) * M));
    //         // checkCudaErrors(cudaMalloc((void **)&plan->histo_count,sizeof(int)*(plan->nf1*plan->nf2+1)));
    //     }
    //     case 2:
    //     {
    //         //multi pass
    //     }
    //     break;

    //     default:
    //         std::cerr << "err: invalid method " << std::endl;
    // }
    if(plan->opts.gpu_gridder_method){plan->hivesize[0]=8;plan->hivesize[1]=8;plan->hivesize[2]=8;};
    if(plan->opts.gpu_gridder_method==5){plan->hivesize[0]=1;plan->hivesize[1]=8;plan->hivesize[2]=8;};
    if(plan->opts.gpu_gridder_method==6){plan->hivesize[0]=8;plan->hivesize[1]=8;plan->hivesize[2]=1;};
    // correction factor memory allocation
    if (!plan->opts.gpu_conv_only)
    {
        if (plan->type != 3)
        {
            checkCudaErrors(cudaMalloc(&plan->fwkerhalf1, (plan->nf1 / 2 + 1) * sizeof(PCS)));
            if (plan->dim > 1)
            {
                checkCudaErrors(cudaMalloc(&plan->fwkerhalf2, (plan->nf2 / 2 + 1) * sizeof(PCS)));
            }
            if (plan->dim > 2)
            {
                // printf("I am inn %d\n",plan->nf3);
                checkCudaErrors(cudaMalloc(&plan->fwkerhalf3, (plan->nf3 / 2 + 1) * sizeof(PCS)));
            }
            
        }
        else
        {
            // nupt to nupt
            checkCudaErrors(cudaMalloc(&plan->fwkerhalf1, (plan->ms) * sizeof(PCS)));
            if (plan->dim > 1)
            {
                checkCudaErrors(cudaMalloc(&plan->fwkerhalf2, (plan->mt) * sizeof(PCS)));
            }
            if (plan->dim > 2)
            {
                checkCudaErrors(cudaMalloc(&plan->fwkerhalf3, (plan->mu) * sizeof(PCS)));
            }
        }

        /* For multi GPU
        cudaStream_t* streams =(cudaStream_t*) malloc(plan->opts.gpu_nstreams*
        sizeof(cudaStream_t));
        for(int i=0; i<plan->opts.gpu_nstreams; i++)
        checkCudaErrors(cudaStreamCreate(&streams[i]));
        plan->streams = streams;
        */
    }

    return ier;
}

int cunufft_setting(int N1, int N2, int N3, int M, int kerevalmeth, int method, int direction, PCS tol, PCS sigma, int type, int dim,
                    PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, CURAFFT_PLAN *plan)
{
    /*
        convolution related parameters setting, plan setting and cufft setting
        N - modes number
        M - number of nupts
        kerevalmeth - kernel (mask) function evaluation method
        method - conv method
        direction - conv direction
        tol - epsilon, expect error
        sigma - upsampling factor
        type - NUFT type (1 NUPT->UPT 2 UPT->NUPT 3 NUPT->NUPT)
        dim - dimension (1,2,3)
        d_u - nupt coorinate
        For type 3, the output coorinates are in plan->d_x y z
    */
    int ier = 0;

    plan->opts.gpu_device_id = 0;
    plan->opts.upsampfac = sigma;
    plan->opts.gpu_sort = 0;
    plan->opts.gpu_binsizex = -1;
    plan->opts.gpu_binsizey = -1;
    plan->opts.gpu_binsizez = -1;
    plan->opts.gpu_kerevalmeth = kerevalmeth;
    plan->opts.gpu_conv_only = 0;
    plan->opts.gpu_gridder_method = method;
    plan->type = type;
    plan->dim = dim;
    plan->execute_flow = 1;
    
    int fftsign = (direction > 0) ? 1 : -1;
    plan->iflag = fftsign; 
    plan->batchsize = 1;
    plan->copts.direction = direction; // 1 inverse, 0 forward

    ier = setup_conv_opts(plan->copts, tol, sigma, 1, direction, kerevalmeth); //check the arguements pirange = 1

    if(kerevalmeth==1){
        PCS *h_c0 = (PCS *)malloc(sizeof(PCS)*SEG_ORDER*SHARED_SIZE_SEG);
        taylor_series_approx_factors(h_c0,plan->copts.ES_beta,SHARED_SIZE_SEG,SEG_ORDER,kerevalmeth);
		checkCudaErrors(cudaMalloc((void**)&plan->c0,sizeof(PCS)*SEG_ORDER*SHARED_SIZE_SEG));
		checkCudaErrors(cudaMemcpy(plan->c0,h_c0,sizeof(PCS)*SEG_ORDER*SHARED_SIZE_SEG,cudaMemcpyHostToDevice));
        free(h_c0);
    }
    

    if (ier != 0)
        printf("setup_error\n");

    int nf1 = 1;
    int nf2 = 1;
    int nf3 = 1;
    plan->ms = N1;
    plan->mt = N2;
    plan->mu = N3;
    if (type == 1)
    {
        // set grid size
        switch (dim)
        {
        case 3:
            nf3 = get_num_cells(N3, plan->copts);

        case 2:
            nf2 = get_num_cells(N2, plan->copts);

        case 1:
            nf1 = get_num_cells(N1, plan->copts);
        default:
            break;
        }
    }
    if (type == 3)
    {
        PCS i_max;
        PCS i_min; // input max and min
        PCS o_max, o_min;
        switch (dim)
        {
        case 1:
        {
            // get input and output coordinates center and width
            get_max_min(i_max, i_min, d_u, M);
            plan->ta.i_center[0] = (i_max + i_min) / (PCS)2.0;
            plan->ta.i_half_width[0] = (i_max - i_min) / (PCS)2.0;

            get_max_min(o_max, o_min, plan->d_x, N1);
            plan->ta.o_center[0] = (o_max + o_min) / (PCS)2.0;
            plan->ta.o_half_width[0] = (o_max - o_min) / (PCS)2.0;
            
            // set scaling ratio (gamma), type3 grid size and grid cell length
            set_nhg_type3(plan->ta.o_half_width[0], plan->ta.i_half_width[0], plan->copts, nf1, plan->ta.h[0], plan->ta.gamma[0]);
            // printf("U_width %lf, U_center %lf, X_width %lf, X_center %lf, gamma %lf, nf %d\n",
                   //plan->ta.i_half_width[0], plan->ta.i_center[0], plan->ta.o_half_width[0], plan->ta.o_center[0], plan->ta.gamma[0], plan->nf1);
            
            // u_j to u_j' x_k to x_k' c_j to c_j'
            pre_stage_invoker(plan->ta.i_center, plan->ta.o_center, plan->ta.gamma, plan->ta.h, d_u, NULL, NULL, plan->d_x, NULL, NULL, d_c, M, N1, 1, 1, plan->iflag);
#ifdef DEBUG
            // coordinates and value printing
            PCS *u = (PCS *)malloc(sizeof(PCS) * M);
            cudaMemcpy(u, d_u, sizeof(PCS) * M, cudaMemcpyDeviceToHost);
            for (int i = 0; i < M; i++)
            {
                printf("%lf ", u[i]);
            }
            printf("\n");
            free(u);
            CPX *c = (CPX *)malloc(sizeof(CPX) * M);
            cudaMemcpy(c, d_c, sizeof(CPX) * M, cudaMemcpyDeviceToHost);
            for (int i = 0; i < M; i++)
            {
                printf("%lf ", c[i].real());
            }
            printf("\n");
            free(c);
            PCS *x = (PCS *)malloc(sizeof(PCS) * N1);
            cudaMemcpy(x, plan->d_x, sizeof(PCS) * N1, cudaMemcpyDeviceToHost);
            for (int i = 0; i < N1; i++)
            {
                printf("%lf ", x[i]);
            }
            printf("\n");
            free(x);
#endif
            break;
        }

        default:
            break;
        }
    }

    setup_plan(nf1, nf2, nf3, M, d_u, d_v, d_w, d_c, plan);
    

    // index sort and type 2 setting ignore
    // calculating correction factor
    // type 1 | 2 - fourier series
    if (type != 3)
    {
        fourier_series_appro_invoker(plan->fwkerhalf1, plan->copts, plan->nf1 / 2 + 1, plan->opts.gpu_kerevalmeth);
        if (dim > 1)
            fourier_series_appro_invoker(plan->fwkerhalf2, plan->copts, plan->nf2 / 2 + 1, plan->opts.gpu_kerevalmeth);
        if (dim > 2)
            fourier_series_appro_invoker(plan->fwkerhalf3, plan->copts, plan->nf3 / 2 + 1, plan->opts.gpu_kerevalmeth);
    }
    // type 3 - fourier series
    else
    {
        fourier_series_appro_invoker(plan->fwkerhalf1, plan->d_x, plan->copts, N1, plan->opts.gpu_kerevalmeth);
        // other dim ++++
    }

    // cufft plan setting
    cufftHandle fftplan;
    //need to check and revise (the partial conv will be differnt)
    if (dim == 1)
    {
        int n[] = {plan->nf1};
        int inembed[] = {plan->nf1};
        int onembed[] = {plan->nf1};
        cufftPlanMany(&fftplan, 1, n, inembed, 1, inembed[0],
                      onembed, 1, onembed[0], CUFFT_TYPE, plan->batchsize); // batch issue
    }
    if (dim == 2)
    {
        int n[] = {plan->nf2, plan->nf1};
        int inembed[] = {plan->nf2, plan->nf1};
        int onembed[] = {plan->nf2, plan->nf1};
        cufftPlanMany(&fftplan, 2, n, inembed, 1, inembed[0] * inembed[1],
                      onembed, 1, onembed[0] * onembed[1], CUFFT_TYPE, plan->batchsize);
    }
    if (dim == 3)
    {
        int n[] = {plan->nf3, plan->nf2, plan->nf1};
        int inembed[] = {plan->nf3, plan->nf2, plan->nf1};
        int onembed[] = {plan->nf3, plan->nf2, plan->nf1};
        cufftPlanMany(&fftplan, 3, n, inembed, 1, inembed[0] * inembed[1] * inembed[2],
                      onembed, 1, onembed[0] * onembed[1] * onembed[2], CUFFT_TYPE, plan->batchsize);
    }
    plan->fftplan = fftplan;
    return 0;
}

int cunufft_exec(CURAFFT_PLAN *plan)
{
    // step1
    curafft_conv(plan);
#ifdef DEBUG
    printf("conv result printing (first w plane)...\n");
    CPX *fw = (CPX *)malloc(sizeof(CPX) * plan->nf1 * plan->nf2 * plan->nf3);
    cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
    PCS temp = 0;

    for (int j = 0; j < plan->nf1; j++)
    {
        temp += fw[j].real();
        printf("%.3g ", fw[j].real());
    }
    printf("fft 000 %.3g\n", temp);
#endif
    // +++++++++++++
    return 0;
}

int curafft_free(CURAFFT_PLAN *plan)
{
    /*
    Free device memory
    */
    int ier = 0;

    switch (plan->dim)
    {
    case 3:
        checkCudaErrors(cudaFree(plan->fwkerhalf3));
        checkCudaErrors(cudaFree(plan->d_w));
    case 2:
        checkCudaErrors(cudaFree(plan->fwkerhalf2));
        checkCudaErrors(cudaFree(plan->d_v));
    case 1:
        checkCudaErrors(cudaFree(plan->fwkerhalf1));
        checkCudaErrors(cudaFree(plan->d_u));
        checkCudaErrors(cudaFree(plan->d_c));
        checkCudaErrors(cudaFree(plan->fw));
        if (!plan->opts.gpu_conv_only)
            checkCudaErrors(cudaFree(plan->fk));

    default:
        break;
    }

    return ier;
}

//------------------------------Below this line, all contents are just for Radio astronomy-------------------------
__global__ void pre_stage_1(PCS i_center, PCS *d_z, CUCPX *d_fk, int N1, int N2, PCS xpixelsize, PCS ypixelsize, int flag)
{
    /*
    prestage 1: cj to cj' (e^(i*x_center*u)*cj) caused by shifting
    Input:
        o_center_*: center of u (v,w) coordinate
        d_u - coordinates
        d_c - value
        M - number of input coordinates 
        flag - -1 or 1
    */
    int idx;
    CUCPX temp;
    for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N1*N2; idx += gridDim.x * blockDim.x)
    {
        int col = idx % N1;
        int row = idx / N1;
        int idx_z = abs(col - N1 / 2) + abs(row - N2 / 2) * (N1 / 2 + 1);
        PCS phase = i_center * (sqrt(1.0 - pow((row-N2/2)*xpixelsize,2) - pow((col-N1/2)*ypixelsize,2)) - 1);
        temp.x = d_fk[idx].x * cos(phase * flag) - d_fk[idx].y * sin(phase * flag);
        temp.y = d_fk[idx].x * sin(phase * flag) + d_fk[idx].y * cos(phase * flag);
        d_fk[idx] = temp;
    }
}

void pre_stage_1_invoker(PCS i_center, PCS *d_z, CUCPX *d_fk, int N1, int N2, PCS xpixelsize, PCS ypixelsize, int flag){
    int blocksize = 512;
    if(d_fk!=NULL){
        if (i_center != 0)
        {
            // cj to cj'
            pre_stage_1<<<(N1*N2 - 1) / blocksize + 1, blocksize>>>(i_center, d_z, d_fk, N1, N2, xpixelsize, ypixelsize, flag);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
}



__global__ void w_term_dft(CUCPX *fw, int nf1, int nf2, int nf3, int N1, int N2, PCS *z, int flag, int batchsize)
{
    /*
        Specified for radio astronomy
        W term dft output driven method (idx takes charge of idx's dft in CMCL mode)
        the output of cufft is FFTW format// just do dft on the in range pixels
        Input:
            fw - result after ffts towards each w plane (FFTW mode)
            z - output coordinate after rescaling, size - (N1/2+1,N2/2+1)
            flag - fourier flag
            i|o_center - center of input|output
    */
    int idx;
    flag = 1.0;
    for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N1 * N2; idx += gridDim.x * blockDim.x)
    {
        int row = idx / N1;
        int col = idx % N1;
        unsigned long long int idx_fw = 0;
        int w1 = 0;
        int w2 = 0;

        w1 = col >= N1 / 2 ? col - N1 / 2 : nf1 + col - N1 / 2;
        w2 = row >= N2 / 2 ? row - N2 / 2 : nf2 + row - N2 / 2;
        idx_fw = w1 + w2 * nf1;
        CUCPX temp;
        temp.x = 0;
        temp.y = 0;

        // double z_t_2pi = 2 * PI * (z); w have been scaling to pirange
        // currently not support for partial computing
        int i;
        int idx_z = abs(col - N1 / 2) + abs(row - N2 / 2) * (N1 / 2 + 1);
        // in w axis the fw is 0 to N, not FFTW mode
        unsigned long long int temp_idx = nf1;
        temp_idx *= nf2;
        for (i = 0; i < nf3 / 2; i++)
        {
            temp.x += fw[idx_fw + temp_idx*(i + nf3 / 2)].x * cos(z[idx_z] * i * flag) - fw[idx_fw + temp_idx * (i + nf3 / 2)].y * sin(z[idx_z] * i * flag);
            temp.y += fw[idx_fw + temp_idx*(i + nf3 / 2)].x * sin(z[idx_z] * i * flag) + fw[idx_fw + temp_idx * (i + nf3 / 2)].y * cos(z[idx_z] * i * flag);
        }
        for (; i < nf3; i++)
        {
            temp.x += fw[idx_fw + temp_idx * (i - nf3 / 2)].x * cos(z[idx_z] * (i - nf3) * flag) - fw[idx_fw + temp_idx * (i - nf3 / 2)].y * sin(z[idx_z] * (i - nf3) * flag);
            temp.y += fw[idx_fw + temp_idx * (i - nf3 / 2)].x * sin(z[idx_z] * (i - nf3) * flag) + fw[idx_fw + temp_idx * (i - nf3 / 2)].y * cos(z[idx_z] * (i - nf3) * flag);
        }
        fw[idx_fw] = temp;
    }
}


__global__ void w_term_idft(CUCPX *fw, int nf1, int nf2, int nf3, int N1, int N2, PCS *z, int flag){
    
    int idx; // there is another way to utilize shared memeory
    for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N1 * N2; idx += gridDim.x * blockDim.x)
    {
        // int plane = idx / (nf1 * nf2);
        // int row = (idx / nf1) % nf2;
        // int col = idx % nf1;
        // int plane = idx / (N1 * N2);
        int row = idx / N1;
        int col = idx % N1;
        unsigned long long int idx_fw = 0;
        int w1 = 0;
        int w2 = 0;

        w1 = col >= N1 / 2 ? col - N1 / 2 : nf1 + col - N1 / 2;
        w2 = row >= N2 / 2 ? row - N2 / 2 : nf2 + row - N2 / 2;
        idx_fw = w1 + w2 * nf1;
        CUCPX temp;
        temp.x = 0;
        temp.y = 0;
        
        // double z_t_2pi = 2 * PI * (z); w have been scaling to pirange
        // currently not support for partial computing
        int idx_z = abs(col - N1 / 2) + abs(row - N2 / 2) * (N1 / 2 + 1);
        for(int plane=1; plane<nf3; plane++){
            PCS phase = flag * z[idx_z] * (plane-nf3/2); 
            temp.x = fw[idx_fw].x * cos(phase) - fw[idx_fw].y * sin(phase);
            temp.y = fw[idx_fw].x * sin(phase) + fw[idx_fw].y * cos(phase);
            unsigned long long int other_idx = nf2;
            other_idx *= nf1;
            other_idx *= plane;
            other_idx += idx_fw;
            fw[other_idx] = temp;
        }
        PCS phase = flag * z[idx_z] * (-nf3/2); 
        temp.x = fw[idx_fw].x * cos(phase) - fw[idx_fw].y * sin(phase);
        temp.y = fw[idx_fw].x * sin(phase) + fw[idx_fw].y * cos(phase);
        fw[idx_fw] = temp;

    }
}

void curadft_invoker(CURAFFT_PLAN *plan, PCS xpixelsize, PCS ypixelsize)
{
    /*
        Specified for radio astronomy
        Input: 
            fw - the res after 2D-FT towards each w
        Output:
            fw - after dft (part/whole based on batchsize)
    */
    int nf1 = plan->nf1;
    int nf2 = plan->nf2;
    int nf3 = plan->nf3;
    int N1 = plan->ms;
    int N2 = plan->mt;
    int batchsize = plan->nf3;
    int flag = plan->iflag;
    
    int num_threads = 512;
    if (flag==1){
        dim3 block(num_threads);
        dim3 grid((N1 * N2 - 1) / num_threads + 1);
        w_term_dft<<<grid, block>>>(plan->fw, nf1, nf2, nf3, N1, N2, plan->d_x, flag, batchsize);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    else {
        // PCS i_center = plan->ta.i_center[0];
        // * e(-c*i*(n_lm-1)) // n_lm-1 -> z, c
        dim3 block(num_threads); 
        dim3 grid((N1 * N2 - 1) / num_threads + 1);
        // dim3 grid((N1*N2*nf3-1)/num_threads+1);
        // PCS adt = plan->ta.gamma[0] * plan->ta.h[0] * plan->ta.o_center[0];
        w_term_idft<<<grid,block>>>(plan->fw, nf1, nf2, nf3, N1, N2, plan->d_x, flag);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    return;
}


__global__ void partial_w_term_dft(CUCPX *fw, CUCPX *fw_temp, int nf1, int nf2, int nf3, int N1, int N2, PCS *z, int flag, int plane_id, int batchsize)
{
    /*
        Specified for radio astronomy
        W term dft output driven method (idx takes charge of idx's dft in CMCL mode)
        the output of cufft is FFTW format// just do dft on the in range pixels
        Input:
            fw - result after ffts towards each w plane (FFTW mode)
            z - output coordinate after rescaling, size - (N1/2+1,N2/2+1)
            flag - fourier flag
            i|o_center - center of input|output
    */
    int idx;
    flag = 1.0;
    for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N1 * N2; idx += gridDim.x * blockDim.x)
    {
        int row = idx / N1;
        int col = idx % N1;
        int idx_fw = 0;
        int w1 = 0;
        int w2 = 0;

        w1 = col >= N1 / 2 ? col - N1 / 2 : nf1 + col - N1 / 2;
        w2 = row >= N2 / 2 ? row - N2 / 2 : nf2 + row - N2 / 2;
        idx_fw = w1 + w2 * nf1;
        CUCPX temp;
        temp.x = 0;
        temp.y = 0;

        // double z_t_2pi = 2 * PI * (z); w have been scaling to pirange
        // currently not support for partial computing
        int i;
        int idx_z = abs(col - N1 / 2) + abs(row - N2 / 2) * (N1 / 2 + 1);
        unsigned long long int temp_idx = nf1;
        temp_idx *= nf2;
        
        for(i = 0; i<batchsize; i++){
            temp.x += fw[idx_fw + temp_idx * i].x * cos(z[idx_z] * (i + plane_id - nf3/2) * flag) - fw[idx_fw + temp_idx * i].y * sin(z[idx_z] * (i + plane_id - nf3/2) * flag);
            temp.y += fw[idx_fw + temp_idx * i].x * sin(z[idx_z] * (i + plane_id - nf3/2) * flag) + fw[idx_fw + temp_idx * i].y * cos(z[idx_z] * (i + plane_id - nf3/2) * flag);
        }
        fw_temp[idx_fw].x += temp.x;
        fw_temp[idx_fw].y += temp.y;
    }
}

void curadft_partial_invoker(CURAFFT_PLAN *plan, PCS xpixelsize, PCS ypixelsize, int plane_id){
    int nf1 = plan->nf1;
    int nf2 = plan->nf2;
    int nf3 = plan->mem_limit;
    int N1 = plan->ms;
    int N2 = plan->mt;
    int batchsize = plan->nf3;
    int flag = plan->iflag;
    int num_threads = 512;
    if (flag==1){
        dim3 block(num_threads);
        dim3 grid((N1 * N2 - 1) / num_threads + 1);
        partial_w_term_dft<<<grid, block>>>(plan->fw, plan->fw_temp, nf1, nf2, nf3, N1, N2, plan->d_x, flag, plane_id, batchsize);
        checkCudaErrors(cudaDeviceSynchronize());
    }
}
