#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
//#include <thrust>
using namespace thrust;

#include "conv_interp_invoker.h"
#include "cuft.h"
#include "utils.h"

///conv improved WS, method 0 correctness cheak

int main(int argc, char *argv[])
{

    //gpu_method == 0, nupts driven
    int N1, N2; //N1 width output
    PCS sigma = 2.0;
    int M; // input
    if (argc < 4)
    {
        fprintf(stderr,
                "Usage: conv2d method nupts_distr nf1 nf2 [maxsubprobsize [M [tol [kerevalmeth [sort]]]]]\n"
                "Arguments:\n"
                "  method: One of\n"
                "    0: nupts driven,\n"
                "  N1, N2 : image size.\n"
                "  M: The number of non-uniform points.\n"
                "  tol: tolerance (default 1e-6).\n"
                "  kerevalmeth: Kernel evaluation method; one of\n"
                "     0: Exponential of square root (default), or\n"
                "     1: taylor series approximation.\n");
        return 1;
    }
    //no result
    double w;
    int method;
    sscanf(argv[1], "%d", &method);
    sscanf(argv[2], "%lf", &w);
    N1 = (int)w; // so can read 1e6 right!
    sscanf(argv[3], "%lf", &w);
    N2 = (int)w; // so can read 1e6 right!

    M = N1 * N2;
    if (argc > 4)
    {
        sscanf(argv[4], "%lf", &w);
        M = (int)w; // so can read 1e6 right!
    }

    PCS tol = 1e-6;
    if (argc > 5)
    {
        sscanf(argv[5], "%lf", &w);
        tol = (PCS)w; // so can read 1e6 right!
    }

    int kerevalmeth = 0;
    if (argc > 6)
    {
        sscanf(argv[6], "%d", &kerevalmeth);
    }

    N1 = 5;
    N2 = 5;
    M = 25; //for correctness checking
    //int ier;
    PCS *x, *y, *z;
    CPX *c, *fw;
    x = (PCS *)malloc(M * sizeof(PCS)); //Allocates page-locked memory on the host.
    y = (PCS *)malloc(M * sizeof(PCS));
    c = (CPX *)malloc(M * sizeof(CPX));

    PCS *d_x, *d_y;
    CUCPX *d_c, *d_fw;
    checkCudaErrors(cudaMalloc(&d_x, M * sizeof(PCS)));
    checkCudaErrors(cudaMalloc(&d_y, M * sizeof(PCS)));
    checkCudaErrors(cudaMalloc(&d_c, M * sizeof(CUCPX)));
    //checkCudaErrors(cudaMalloc(&d_fw,8*nf1*nf2*nf1*sizeof(CUCPX)));

    //generating data
    int nupts_distribute = 0;
    switch (nupts_distribute)
    {
    case 0: //uniform
    {
        for (int i = 0; i < M; i++)
        {
            x[i] = M_PI * randm11();
            y[i] = M_PI * randm11();
            c[i].real(1.0); //back to random11()
            c[i].imag(1.0);
        }
    }
    break;
    case 1: // concentrate on a small region
    {
        for (int i = 0; i < M; i++)
        {
            x[i] = M_PI * rand01() / N1 * 16;
            y[i] = M_PI * rand01() / N2 * 16;
            z[i] = M_PI * rand01() / N2 * 16;
            c[i].real(randm11());
            c[i].imag(randm11());
        }
    }
    break;
    default:
        std::cerr << "not valid nupts distr" << std::endl;
        return 1;
    }
    double a[5] = {-PI / 2, -PI / 3, 0, PI / 3, PI / 2};
    for (int i = 0; i < 25; i++)
    {
        x[i] = a[i / 5];
        y[i] = a[i % 5];
    }

    //printf("generated data, x[1] %2.2g, y[1] %2.2g , z[1] %2.2g, c[1] %2.2g\n",x[1] , y[1], z[1], c[1].real());
    //data transfer
    checkCudaErrors(cudaMemcpy(d_x, x, M * sizeof(PCS), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y, M * sizeof(PCS), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_c, c, M * sizeof(CUCPX), cudaMemcpyHostToDevice));

    CURAFFT_PLAN *h_plan = new CURAFFT_PLAN();
    memset(h_plan, 0, sizeof(CURAFFT_PLAN));

    // opts and copts setting
    h_plan->opts.gpu_conv_only = 1;
    h_plan->opts.gpu_gridder_method = method;
    h_plan->opts.gpu_kerevalmeth = kerevalmeth;
    h_plan->opts.gpu_sort = 1;
    h_plan->opts.upsampfac = sigma;
    h_plan->dim = 2;
    // h_plan->copts.pirange = 1;
    // some plan setting

    int ier = setup_conv_opts(h_plan->copts, tol, sigma, 1, 1, kerevalmeth); //check the arguements

    if (ier != 0)
        printf("setup_error\n");

    // plan setting
    int nf1 = (int)sigma * N1;
    int nf2 = (int)sigma * N2;
    int nf3 = 1;
    ier = setup_plan(nf1, nf2, nf3, M, d_x, d_y, NULL, d_c, h_plan); //cautious the number of plane using N1 N2 to get nf1 nf2

    //printf("the num of w %d\n",h_plan->num_w);

    // printf("the kw is %d\n", h_plan->copts.kw);
    int f_size = nf1 * nf2 * nf3;
    fw = (CPX *)malloc(sizeof(CPX) * f_size);
    checkCudaErrors(cudaMalloc(&d_fw, f_size * sizeof(CUCPX)));

    h_plan->fw = d_fw;
    //checkCudaErrors(cudaMallocHost(&fw,nf1*nf2*h_plan->num_w*sizeof(CPX))); //malloc after plan setting
    //checkCudaErrors(cudaMalloc( &d_fw,( nf1*nf2*(h_plan->num_w)*sizeof(CUCPX) ) ) ); //check

    std::cout << std::scientific << std::setprecision(3); //setprecision not define

    cudaEvent_t cuda_start, cuda_end;

    float kernel_time;

    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);

    cudaEventRecord(cuda_start);

    // convolution
    curafft_conv(h_plan); //add to include
    cudaEventRecord(cuda_end);

    cudaEventSynchronize(cuda_start);
    cudaEventSynchronize(cuda_end);

    cudaEventElapsedTime(&kernel_time, cuda_start, cuda_end);

    // checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(fw, d_fw, sizeof(CUCPX) * f_size, cudaMemcpyDeviceToHost));

    //int nf3 = h_plan->num_w;
    printf("Method %d (nupt driven) %d NU pts to #%d U pts in %.3g s\n",
           h_plan->opts.gpu_gridder_method, M, nf1 * nf2 * nf3, kernel_time / 1000);

    curafft_free(h_plan);

    std::cout << "[result-input]" << std::endl;
    for (int k = 0; k < nf3; k++)
    {
        for (int j = 0; j < nf2; j++)
        {
            for (int i = 0; i < nf1; i++)
            {
                printf(" (%2.3g,%2.3g)", fw[i + j * nf1 + k * nf2 * nf1].real(),
                       fw[i + j * nf1 + k * nf2 * nf1].imag());
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;

    checkCudaErrors(cudaDeviceReset());
    free(x);
    free(y);
    //free(z);
    free(c);
    free(fw);

    return 0;
}