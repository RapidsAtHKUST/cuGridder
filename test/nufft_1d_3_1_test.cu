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
    int N = 100;
    PCS sigma = 2; // upsampling factor
    int M = 100;

    PCS epsilon = 1e-6;

    int kerevalmeth = 0;

    int method = 0;

    //gpu_method == 0, nupts driven

    //int ier;
    PCS *u;
    CPX *c;
    u = (PCS *)malloc(M * sizeof(PCS)); //Allocates page-locked memory on the host.
    c = (CPX *)malloc(M * sizeof(CPX));
    PCS *d_u;
    CUCPX *d_c, *d_fk;
    CUCPX *d_fw;
    checkCudaErrors(cudaMalloc(&d_u, M * sizeof(PCS)));
    checkCudaErrors(cudaMalloc(&d_c, M * sizeof(CUCPX)));
    /// pixel size
    // generating data
    for (int i = 0; i < M; i++)
    {
        u[i] = 2.0 + i * PI / (double)M; //xxxxx
        c[i].real(randm11() * 1000);
        c[i].imag(i);
        // wgt[i] = 1;
    }

    PCS *k = (PCS *)malloc(sizeof(PCS) * N * 10);
    // PCS pixelsize = 0.01;
    for (int i = 0; i < N; i++)
    {
        /* code */
        // k[i] = (int)i-N/2;
        k[i] = -(double)i / (double)N;
        // k[i] = i/(double)N;
        // k[i] = i-N/2 + randm11();
        printf("%.10lf ", k[i]);
    }
    printf("\n");

    //data transfer
    checkCudaErrors(cudaMemcpy(d_u, u, M * sizeof(PCS), cudaMemcpyHostToDevice)); //u
    checkCudaErrors(cudaMemcpy(d_c, c, M * sizeof(CUCPX), cudaMemcpyHostToDevice));

    /* ----------Step2: plan setting------------*/
    CURAFFT_PLAN *plan;

    plan = new CURAFFT_PLAN();
    memset(plan, 0, sizeof(CURAFFT_PLAN));

    PCS *d_k;
    checkCudaErrors(cudaMalloc((void **)&d_k, sizeof(PCS) * N));
    checkCudaErrors(cudaMemcpy(d_k, k, sizeof(PCS) * N, cudaMemcpyHostToDevice));
    plan->d_x = d_k;
    int direction = 1; //inverse

    cunufft_setting(N, 1, 1, M, kerevalmeth, method, direction, epsilon, sigma, 3, 1, d_u, NULL, NULL, d_c, plan);
    int nf1 = plan->nf1;
    printf("conv info printing, sigma %lf, kw %d, beta %lf, nf1 %d\n", plan->copts.upsampfac, plan->copts.kw, plan->copts.ES_beta, nf1);

    // // fourier_series_appro_invoker(d_fwkerhalf,plan->copts,nf1/2+1);
    PCS *fwkerhalf = (PCS *)malloc(sizeof(PCS) * (N));
    checkCudaErrors(cudaMemcpy(fwkerhalf, plan->fwkerhalf1, sizeof(PCS) * (N), cudaMemcpyDeviceToHost));

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
    // fw (conv res set)
    checkCudaErrors(cudaMalloc((void **)&d_fw, sizeof(CUCPX) * plan->nf1));
    checkCudaErrors(cudaMemset(d_fw, 0, sizeof(CUCPX) * plan->nf1));
    plan->fw = d_fw;
    // fk malloc and set
    checkCudaErrors(cudaMalloc((void **)&d_fk, sizeof(CUCPX) * N));
    plan->fk = d_fk;

    // calulating result
    curafft_conv(plan);
    CPX *fw = (CPX *)malloc(sizeof(CPX) * plan->nf1);
    cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1, cudaMemcpyDeviceToHost);
#ifdef DEBUG
    printf("conv result printing...\n");

    for (int i = 0; i < nf1; i++)
    {
        printf("%lf ", fw[i].real());
    }
    printf("\n");

#endif
    PCS *kp = (PCS *)malloc(sizeof(PCS) * N);
    checkCudaErrors(cudaMemcpy(kp, plan->d_x, sizeof(PCS) * N, cudaMemcpyDeviceToHost));

    CPX *fk = (CPX *)malloc(sizeof(CPX) * N);
    memset(fk, 0, sizeof(CPX) * N);
    // dft
    for (int i = 0; i < N; i++)
    {
        /* code */
        for (int j = 0; j < plan->nf1; j++)
        {
            if (j < nf1 / 2)
                fk[i] += fw[j + nf1 / 2] * exp((PCS)j * kp[i] * IMA);
            else
                fk[i] += fw[j - nf1 / 2] * exp(((PCS)j - (PCS)nf1) * kp[i] * IMA); // does j need to change?
                                                                                   // decompose those calculation in order to reach better precision
                                                                                   // double temp1;
                                                                                   // int idx = j + plan->nf1/2;
                                                                                   // temp1 = (double)j/(double)nf1;
                                                                                   // if(j>=nf1/2){
                                                                                   // 	temp1 = temp1 - 1.00000000;
                                                                                   // 	idx -= nf1;
                                                                                   // }
                                                                                   // temp1 *=PI * 2.0000000000;
                                                                                   // temp1 *= k[i];
            // fk[i] = fk[i] + fw[idx]*exp((double)temp1*IMA);

            //
            // fk[i].real( temp2 );
            // fk[i].imag( temp3 );
            // if(j<nf1/2){
            //     fk[i] += fw[j+nf1/2]*exp(k[i]*(((PCS)j)/((PCS)nf1)*2.0*PI*IMA));
            // }
            // else{
            //     fk[i] += fw[j-nf1/2]*exp(k[i]*((j-nf1)/((PCS)nf1) )*2.0*PI*IMA); // decompose
            // }
        }
    }
#ifdef DEBUG
    printf("dft result printing...\n");
    for (int i = 0; i < N; i++)
    {
        /* code */
        printf("%lf ", fk[i].real());
    }
    printf("\n");
#endif

    // printf("correction factor printing...\n");
    // for(int i=0; i<N1/2; i++){
    // 	printf("%.3g ",fwkerhalf1[i]);
    // }
    // printf("\n");
    // for(int i=0; i<N2/2; i++){
    // 	printf("%.3g ",fwkerhalf2[i]);
    // }
    // printf("\n");
    // deconv
    //PCS *fwkerhalf = (PCS *)malloc(sizeof(PCS)*(N));
    //cudaMemcpy(fwkerhalf, d_fwkerhalf, sizeof(PCS)*(N), cudaMemcpyDeviceToHost);
    printf("i center %lf, o center %lf\n", plan->ta.i_center[0], plan->ta.o_center[0]);
    for (int i = 0; i < N; i++)
    {
        fk[i] = fk[i] / fwkerhalf[i] * exp((k[i] - plan->ta.o_center[0]) * plan->ta.i_center[0] * IMA);
    }

    // result printing

    printf("final result printing...\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.10lf ", fk[i].real());
    }
    printf("\n");
    CPX *truth = (CPX *)malloc(sizeof(CPX) * N);
    printf("ground truth printing...\n");
    for (int i = 0; i < N; i++)
    {
        truth[i] = 0;
        for (int j = 0; j < M; j++)
        {
            truth[i] += c[j] * exp(k[i] * u[j] * IMA);
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
    for (int i = 0; i < N; i++)
    {
        diff = truth[i] - fk[i];
        err += real(conj(diff) * diff);
        nrm += real(conj(fk[i]) * fk[i]);
    }
    printf("l2 error %.6g\n", sqrt(err / nrm));

    //free
    curafft_free(plan);
    free(fk);
    free(u);
    free(c);

    return ier;
}