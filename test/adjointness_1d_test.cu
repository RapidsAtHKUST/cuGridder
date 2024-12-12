// adjointness testing
#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
//#include <thrust>

#include "ragridder_plan.h"
#include "conv_interp_invoker.h"
#include "cuft.h"
#include "deconv.h"
#include "cugridder.h"
#include "precomp.h"
#include "utils.h"

int main(int argc, char *argv[])
{
    int N = 1;
    PCS sigma = 2; // upsampling factor
    int M = 100;
    PCS epsilon = 1e-12;
    int kerevalmeth = 0;
    int method = 0;

    // memory allocation
    //Host
    PCS *u;
    CPX *c;
    CPX *fw;
    PCS *zp;
    PCS *fwkerhalf;
    u = (PCS *)malloc(M * sizeof(PCS));
    c = (CPX *)malloc(M * sizeof(CPX));
    PCS *z = (PCS *)malloc(N * sizeof(PCS));
    CPX *fk = (CPX *)malloc(N * sizeof(CPX));
    //Device
    PCS *d_u, *d_z;
    CUCPX *d_c, *d_fk;
    CUCPX *d_fw;
    checkCudaError(cudaMalloc(&d_u, M * sizeof(PCS)));
    checkCudaError(cudaMalloc(&d_z, N * sizeof(PCS)));
    checkCudaError(cudaMalloc(&d_c, M * sizeof(CUCPX)));
    checkCudaError(cudaMalloc(&d_fk, N * sizeof(CUCPX)));

    // data generation
    for (int i = 0; i < M; i++)
    {
        u[i] = 2.0 + i; //xxxxx
        c[i].real(randm11());
        c[i].imag(randm11());

        // wgt[i] = 1;
    }

    for (int i = 0; i < N; i++)
    {
        z[i] = -randm11();
        fk[i].real(randm11() * 1000);
        fk[i].imag(0);
    }

    // result allocation
    CPX *c_2 = (CPX *)malloc(M * sizeof(CPX));
    memset(c_2, 0, sizeof(CPX) * M);
    CPX *fk_2 = (CPX *)malloc(N * sizeof(CPX));
    memset(fk_2, 0, sizeof(CPX) * N);

    /*-------------------------------------- C -> Fk ---------------------------------*/
    // setting plan
    CURAFFT_PLAN *plan;
    plan = new CURAFFT_PLAN();
    memset(plan, 0, sizeof(CURAFFT_PLAN));

    // memory transfering
    checkCudaError(cudaMemcpy(d_z, z, sizeof(PCS) * N, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_u, u, sizeof(PCS) * M, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_c, c, sizeof(CUCPX) * M, cudaMemcpyHostToDevice));

    plan->d_x = d_z;
    int direction = 1;

    cunufft_setting(N, 1, 1, M, kerevalmeth, method, direction, epsilon, sigma, 3, 1, d_u, NULL, NULL, d_c, plan);
    int nf1 = plan->nf1;

    // correction factor
    fwkerhalf = (PCS *)malloc(sizeof(PCS) * (N));
    checkCudaError(cudaMemcpy(fwkerhalf, plan->fwkerhalf1, sizeof(PCS) * (N), cudaMemcpyDeviceToHost));

    // fw malloc and set
    checkCudaError(cudaMalloc((void **)&d_fw, sizeof(CUCPX) * plan->nf1));
    checkCudaError(cudaMemset(d_fw, 0, sizeof(CUCPX) * plan->nf1));
    plan->fw = d_fw;

    CUCPX *d_fk_2;
    checkCudaError(cudaMalloc((void **)&d_fk_2, sizeof(CUCPX) * N));
    checkCudaError(cudaMemset(d_fk_2, 0, sizeof(CUCPX) * N));
    plan->fk = d_fk_2;

    // conv
    curafft_conv(plan);
    fw = (CPX *)malloc(sizeof(CPX) * plan->nf1);
    cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1, cudaMemcpyDeviceToHost);

    zp = (PCS *)malloc(sizeof(PCS) * N);
    checkCudaError(cudaMemcpy(zp, plan->d_x, sizeof(PCS) * N, cudaMemcpyDeviceToHost));

    // dft
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < plan->nf1; j++)
        {
            if (j < nf1 / 2)
                fk_2[i] += fw[j + nf1 / 2] * exp((PCS)j * zp[i] * IMA);
            else
                fk_2[i] += fw[j - nf1 / 2] * exp(((PCS)j - (PCS)nf1) * zp[i] * IMA);
        }
    }

    //deconv
    for (int i = 0; i < N; i++)
    {
        fk_2[i] = fk_2[i] / fwkerhalf[i] * exp((z[i] - plan->ta.o_center[0]) * plan->ta.i_center[0] * IMA);
    }

    CPX *truth = (CPX *)malloc(sizeof(CPX) * N);
    printf("ground truth printing...\n");
    for (int i = 0; i < N; i++)
    {
        truth[i] = 0;
        for (int j = 0; j < M; j++)
        {
            truth[i] += c[j] * exp(z[i] * u[j] * IMA);
        }
    }
    CPX diff;
    double err = 0;
    double nrm = 0;
    for (int i = 0; i < N; i++)
    {
        diff = truth[i] - fk_2[i];
        err += real(conj(diff) * diff);
        nrm += real(conj(fk_2[i]) * fk_2[i]);
    }
    printf("l2 error %.6g\n", sqrt(err / nrm));
    // free
    free(plan);

    /*-------------------------------------- Fk -> C ---------------------------------*/
    plan = new CURAFFT_PLAN();
    memset(plan, 0, sizeof(CURAFFT_PLAN));

    checkCudaError(cudaMemcpy(d_z, z, sizeof(PCS) * N, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_u, u, sizeof(PCS) * M, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemset(d_c, 0, sizeof(CUCPX) * M));

    plan->d_x = d_z;
    direction = 0;

    cunufft_setting(N, 1, 1, M, kerevalmeth, method, direction, epsilon, sigma, 3, 1, d_u, NULL, NULL, d_c, plan);
    nf1 = plan->nf1;
    memset(fw, 0, sizeof(CPX) * nf1);

    checkCudaError(cudaMemcpy(fwkerhalf, plan->fwkerhalf1, sizeof(PCS) * (N), cudaMemcpyDeviceToHost)); // can remove this line

    CPX *fk_1 = (CPX *)malloc(sizeof(CPX) * N);
    // deconv
    for (int i = 0; i < N; i++)
    {
        fk_1[i] = fk[i];
        fk_1[i] = fk_1[i] / fwkerhalf[i] * exp(-(z[i] - plan->ta.o_center[0]) * plan->ta.i_center[0] * IMA);
        // fk[i] = fk[i] / fwkerhalf[i] * exp(-k[i]*plan->ta.i_center[0]*IMA);
    }

    //
    checkCudaError(cudaMemcpy(zp, plan->d_x, sizeof(PCS) * N, cudaMemcpyDeviceToHost));

    // idft
    for (int j = 0; j < plan->nf1; j++)
    {
        for (int i = 0; i < N; i++)
        {
            CPX temp = exp(-(j - nf1 / 2) * zp[i] * IMA);
            fw[j] += fk_1[i] * temp;
        }
    }

    // interp
    checkCudaError(cudaMemcpy(d_fw, fw, sizeof(CUCPX) * plan->nf1, cudaMemcpyHostToDevice));
    plan->fw = d_fw;
    curafft_interp(plan);

    checkCudaError(cudaMemcpy(c_2, plan->d_c, sizeof(CUCPX) * M, cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; i++)
    {
        CPX temp = exp(-u[i] * plan->ta.o_center[0] * IMA);
        c_2[i] = c_2[i] * temp; // some issues
    }

    truth = (CPX *)malloc(sizeof(CPX) * M);
    printf("ground truth printing...\n");
    for (int j = 0; j < M; j++)
    {
        truth[j] = 0;
        for (int i = 0; i < N; i++)
        {
            truth[j] += fk[i] * exp(-z[i] * u[j] * IMA);
        }
    }

    err = 0;
    nrm = 0;
    for (int i = 0; i < M; i++)
    {
        diff = truth[i] - c_2[i];
        err += real(conj(diff) * diff);
        nrm += real(conj(c_2[i]) * c_2[i]);
    }
    printf("l2 error %.6g\n", sqrt(err / nrm));

    PCS adjt_1 = 0;
    for (int i = 0; i < M; i++)
    {
        adjt_1 += (conj(c_2[i]) * c[i]).real();
    }
    PCS adjt_2 = 0;
    for (int i = 0; i < N; i++)
    {
        adjt_2 += (conj(fk_2[i]) * fk[i]).real();
    }
    printf("adjointness checking...\n %.10lf, %.10lf\n", adjt_1, adjt_2);
    //free memory
    // Device
    curafft_free(plan);
    //Host
    free(u);
    free(c);
    free(z);
    free(fk);
    free(c_2);
    free(fk_2);
    free(fwkerhalf);
    free(fw);
    free(zp);
    free(fk_1);

    return 0;
}