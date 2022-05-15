#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
//#include <thrust>
using namespace thrust;

#include "ragridder_plan.h"
#include "conv_interp_invoker.h"
#include "cuft.h"
#include "deconv.h"
#include "cugridder.h"
#include "precomp.h"
#include "utils.h"

// 1d adjointness testing and 1d idft testing

int main(int argc, char *argv[])
{
    /* Input: M, N1, N2, epsilon method
		method - conv method
		M - number of randomly distributed points
		N1, N2 - output size
		epsilon - tolerance
	*/

    // issue related to accuary - how to set sigma, epsilon, number of plane, beta and kw. the number of w plane may need to increase.
    int ier = 0;
    int N = 1;
    PCS sigma = 2; // upsampling factor
    int M = 25;

    PCS epsilon = 1e-12;

    int kerevalmeth = 0;

    int method = 0;

    //gpu_method == 0, nupts driven

    //int ier;
    PCS *u;
    CPX *c;
    u = (PCS *)malloc(M * sizeof(PCS));

    CPX *fk = (CPX *)malloc(N * sizeof(CPX));
    PCS *d_u;
    CUCPX *d_c, *d_fk;
    CUCPX *d_fw;
    checkCudaError(cudaMalloc(&d_u, M * sizeof(PCS)));
    checkCudaError(cudaMalloc(&d_fk, N * sizeof(CUCPX)));
    checkCudaError(cudaMalloc(&d_c, M * sizeof(CUCPX)));
    /// pixel size
    // generating data
    for (int i = 0; i < M; i++)
    {
        u[i] = 2.0 + i; //xxxxx

        // wgt[i] = 1;
    }

    PCS *k = (PCS *)malloc(sizeof(PCS) * N * 10);
    // PCS pixelsize = 0.01;
    for (int i = 0; i < N; i++)
    {
        /* code */
        // k[i] = (int)i-N/2;
        k[i] = -(double)i / (double)N;
        fk[i].real(randm11() * 1000);
        fk[i].imag(0);
        // k[i] = i/(double)N;
        // k[i] = i-N/2 + randm11();
    }
    printf("\n");

    //data transfer
    checkCudaError(cudaMemcpy(d_u, u, M * sizeof(PCS), cudaMemcpyHostToDevice)); //u

    /* ----------Step2: plan setting------------*/
    CURAFFT_PLAN *plan;

    plan = new CURAFFT_PLAN();
    memset(plan, 0, sizeof(CURAFFT_PLAN));

    PCS *d_k;
    checkCudaError(cudaMalloc((void **)&d_k, sizeof(PCS) * N));
    checkCudaError(cudaMemcpy(d_k, k, sizeof(PCS) * N, cudaMemcpyHostToDevice));
    plan->d_x = d_k;
    int direction = 0;

    cunufft_setting(N, 1, 1, M, kerevalmeth, method, direction, epsilon, sigma, 3, 1, d_u, NULL, NULL, d_c, plan);
    int nf1 = plan->nf1;
    //printf("conv info printing, sigma %lf, kw %d, beta %lf, nf1 %d\n", plan->copts.upsampfac, plan->copts.kw, plan->copts.ES_beta, nf1);

    // // fourier_series_appro_invoker(d_fwkerhalf,plan->copts,nf1/2+1);
    PCS *fwkerhalf = (PCS *)malloc(sizeof(PCS) * (N));
    checkCudaError(cudaMemcpy(fwkerhalf, plan->fwkerhalf1, sizeof(PCS) * (N), cudaMemcpyDeviceToHost));

    //fourier_series(fwkerhalf,k,plan->copts,N,nf1/2+1);
#ifdef DEBUG
    printf("correction factor printing method1...\n");
    for (int i = 0; i < N; i++)
    {
        /* code */
        printf("%lf ", fwkerhalf[i]);
    }
    printf("\n");
#endif
    CPX *fk_1 = (CPX *)malloc(sizeof(CPX) * N);
    // deconv
    for (int i = 0; i < N; i++)
    {
        fk_1[i] = fk[i];
        fk[i] = fk[i] / fwkerhalf[i] * exp(-(k[i] - plan->ta.o_center[0]) * plan->ta.i_center[0] * IMA);
        // fk[i] = fk[i] / fwkerhalf[i] * exp(-k[i]*plan->ta.i_center[0]*IMA);
    }
#ifdef DEBUG
    for (int i = 0; i < N; i++)
    {
        printf("<%.6g,%.6g> ", fk[i].real(), fk[i].imag());
    }
    printf("\n");
#endif

    PCS *kp = (PCS *)malloc(sizeof(PCS) * N);
    checkCudaError(cudaMemcpy(kp, plan->d_x, sizeof(PCS) * N, cudaMemcpyDeviceToHost));

    // idft
    CPX *fw = (CPX *)malloc(sizeof(CPX) * plan->nf1);
    memset(fw, 0, sizeof(CPX) * plan->nf1);
    for (int j = 0; j < plan->nf1; j++)
    {
        for (int i = 0; i < N; i++)
        {
            CPX temp = exp(-(j - nf1 / 2) * kp[i] * IMA);
            fw[j] += fk[i] * temp;
        }
    }

#ifdef DEBUG
    printf("idft result printing...\n");
    for (int i = 0; i < nf1; i++)
    {
        printf("<%.6g,%.6g> ", fw[i].real(), fw[i].imag());
    }
    printf("\n");
#endif
    // fw (conv res set)
    checkCudaError(cudaMalloc((void **)&d_fw, sizeof(CUCPX) * plan->nf1));
    checkCudaError(cudaMemcpy(d_fw, fw, sizeof(CUCPX) * plan->nf1, cudaMemcpyHostToDevice));
    plan->fw = d_fw;

    // calulating result
    curafft_interp(plan);
    c = (CPX *)malloc(sizeof(CPX) * M);
    checkCudaError(cudaMemcpy(c, plan->d_c, sizeof(CUCPX) * M, cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; i++)
    {
        CPX temp = exp(-u[i] * plan->ta.o_center[0] * IMA);
        c[i] = c[i] * temp; // some issues
    }

    // result printing

    printf("final result printing...\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.10lf ", c[i].real());
    }
    printf("\n");
    CPX *truth = (CPX *)malloc(sizeof(CPX) * M);
    printf("ground truth printing...\n");
    for (int j = 0; j < M; j++)
    {
        truth[j] = 0;
        for (int i = 0; i < N; i++)
        {
            truth[j] += fk_1[i] * exp(-k[i] * u[j] * IMA);
        }
    }

    for (int i = 0; i < 10; i++)
    {
        printf("%.10lf ", truth[i].real());
    }
    printf("\n");

    // double fk_max = 0;
    // for(int i=0; i<M; i++){
    // 	if(abs(fk[i].real())>fk_max)fk_max = abs(fk[i].real());
    // }
    // printf("fk max %lf\n",fk_max);
    CPX diff;
    double err = 0;
    double nrm = 0;
    for (int i = 0; i < M; i++)
    {
        diff = truth[i] - c[i];
        err += real(conj(diff) * diff);
        nrm += real(conj(c[i]) * c[i]);
    }
    printf("l2 error %.6g\n", sqrt(err / nrm));

    //free
    curafft_free(plan);
    free(fk);
    free(fw);
    free(k);
    free(kp);
    free(fwkerhalf);
    free(fk_1);
    free(truth);
    free(u);
    free(c);

    return ier;
}