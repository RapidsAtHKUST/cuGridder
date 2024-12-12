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
#include "cugridder.h"
#include "precomp.h"
#include "utils.h"

///conv improved WS, method 0 correctness cheak

int main(int argc, char *argv[])
{
    // suppose there is just one channel
    // range of uvw [-lamda/2,lamda/2]， rescale with factor resolution / fov compatible with l
    // l and m need to be converted into pixels
    /* Input: nrow, nchan, nxdirty, nydirty, fov, epsilon
		row - number of visibility
		nchan - number of channels
		nxdirty, nydirty - image size (height width)
		fov - field of view
		epsilon - tolerance
	*/
    int ier = 0;
    if (argc < 7)
    {
        fprintf(stderr,
                "Usage: degridding, dirty image to visibility\n"
                "Arguments:\n"
                "  method: One of\n"
                "    0: upt driven,\n"
                "  nxdirty, nydirty : image size.\n"
                "  nrow: The number of non-uniform points.\n"
                "  fov: Field of view.\n"
                "  nchan: number of chanels (default 1)\n"
                "  epsilon: NUFFT tolerance (default 1e-12).\n"
                "  kerevalmeth: Kernel evaluation method; one of\n"
                "     0: Exponential of square root (default), or\n"
                "     1: Taylor t evaluation.\n");
        return 1;
    }
    int nxdirty, nydirty;
    PCS sigma = 2; // upsampling factor
    int nrow, nchan;
    PCS fov;

    double inp;
    int method;
    sscanf(argv[1], "%d", &method);
    int w_term_method;
    sscanf(argv[2], "%d", &w_term_method);
    sscanf(argv[3], "%d", &nxdirty);
    sscanf(argv[4], "%d", &nydirty);
    sscanf(argv[5], "%d", &nrow);
    sscanf(argv[6], "%lf", &inp);
    fov = inp;

    nchan = 1;
    if (argc > 7)
    {
        sscanf(argv[7], "%d", &nchan);
    }

    PCS epsilon = 1e-12;
    if (argc > 8)
    {
        sscanf(argv[8], "%lf", &inp);
        epsilon = (PCS)inp; // so can read 1e6 right!
    }

    int kerevalmeth = 0;
    if (argc > 9)
    {
        sscanf(argv[9], "%d", &kerevalmeth);
    }

    // degree per pixel (unit radius)
    // PCS deg_per_pixelx = fov / 180.0 * PI / (PCS)nxdirty;
    // PCS deg_per_pixely = fov / 180.0 * PI / (PCS)nydirty;
    // chanel setting
    PCS f0 = 1e9;
    PCS *freq = (PCS *)malloc(sizeof(PCS) * nchan);
    for (int i = 0; i < nchan; i++)
    {
        freq[i] = f0 + i / (double)nchan * fov; //!
    }
    //improved WS stacking 1,
    //gpu_method == 0, nupts driven

    //N1 = 5; N2 = 5; M = 25; //for correctness checking
    //int ier;
    PCS *u, *v, *w;
    CPX *vis;
    PCS *wgt = NULL;                       //currently no mask
    u = (PCS *)malloc(nrow * sizeof(PCS)); //Allocates page-locked memory on the host.
    v = (PCS *)malloc(nrow * sizeof(PCS));
    w = (PCS *)malloc(nrow * sizeof(PCS));
    vis = (CPX *)malloc(nrow * sizeof(CPX));
    PCS *d_u, *d_v, *d_w;
    CUCPX *d_vis, *d_fk;

    CPX *dirty_image = (CPX *)malloc(sizeof(CPX) * nxdirty * nydirty);

    checkCudaError(cudaMalloc((void **)&d_u, nrow * sizeof(PCS)));
    checkCudaError(cudaMalloc((void **)&d_v, nrow * sizeof(PCS)));
    checkCudaError(cudaMalloc((void **)&d_w, nrow * sizeof(PCS)));
    checkCudaError(cudaMalloc((void **)&d_vis, nrow * sizeof(CUCPX)));

    PCS pixelsize = fov * PI / 180 / nxdirty;
    // generating data
    for (int i = 0; i < nrow; i++)
    {
        u[i] = randm11() * 0.5 * SPEEDOFLIGHT / f0 / pixelsize; //xxxxx remove
        v[i] = randm11() * 0.5 * SPEEDOFLIGHT / f0 / pixelsize;
        w[i] = randm11() * 0.5 * SPEEDOFLIGHT / f0 * 20000;
        vis[i].real(0); // nrow vis per channel, weight?
        vis[i].imag(0);
        // wgt[i] = 1;
    }
    for (int i = 0; i < nxdirty * nydirty; i++)
    {
        dirty_image[i].real(randm11() * 100);
        dirty_image[i].imag(randm11() * 100);
    }
#ifdef DEBUG
    printf("origial input data...\n");
    for (int i = 0; i < nrow; i++)
    {
        printf("%.12lf ", w[i]);
    }
    printf("\n");
    for (int i = 0; i < nxdirty * nydirty; i++)
    {
        printf("%.12lf ", dirty_image[i].real());
        printf("%.12lf ", dirty_image[i].imag());
    }
    printf("\n");
#endif
    // ignore the tdirty

    // Timing begin ++++
    //data transfer
    checkCudaError(cudaMemcpy(d_u, u, nrow * sizeof(PCS), cudaMemcpyHostToDevice)); //u
    checkCudaError(cudaMemcpy(d_v, v, nrow * sizeof(PCS), cudaMemcpyHostToDevice)); //v
    checkCudaError(cudaMemcpy(d_w, w, nrow * sizeof(PCS), cudaMemcpyHostToDevice)); //w
    checkCudaError(cudaMemcpy(d_vis, vis, nrow * sizeof(CUCPX), cudaMemcpyHostToDevice));

    /* -----------Step1: Baseline setting--------------
	skip negative v
    uvw, nrow = M, shift, mask, f_over_c (fixed due to single channel)
    */
    int shift = 0;
    while ((int(1) << shift) < nchan)
        ++shift;
    // int mask = (int(1) << shift) - 1; // ???
    PCS *f_over_c = (PCS *)malloc(sizeof(PCS) * nchan);
    for (int i = 0; i < nchan; i++)
    {
        f_over_c[i] = freq[i] / SPEEDOFLIGHT;
    }

    /* ----------Step2: cugridder------------*/
    // plan setting
    CURAFFT_PLAN *plan;

    ragridder_plan *gridder_plan;

    plan = new CURAFFT_PLAN();
    gridder_plan = new ragridder_plan();
    memset(plan, 0, sizeof(CURAFFT_PLAN));
    memset(gridder_plan, 0, sizeof(ragridder_plan));

    visibility *pointer_v;
    pointer_v = (visibility *)malloc(sizeof(visibility));
    pointer_v->u = u;
    pointer_v->v = v;
    pointer_v->w = w;
    pointer_v->vis = vis;
    pointer_v->frequency = freq;
    pointer_v->weight = wgt;
    pointer_v->pirange = 0;
    pointer_v->sign = -1;

    int direction = 0; //dirty to vis

    // device data allocation and transfer should be done in gridder setting
    ier = gridder_setting(nydirty, nxdirty, method, kerevalmeth, w_term_method, epsilon, direction, sigma, 0, 1, nrow, nchan, fov, pointer_v, d_u, d_v, d_w, d_vis, plan, gridder_plan);

    //print the setting result
    free(pointer_v);
    if (ier == 1)
    {
        printf("errors in gridder setting\n");
        return ier;
    }
    // fk(image) malloc and set
    checkCudaError(cudaMalloc((void **)&d_fk, sizeof(CUCPX) * nydirty * nxdirty));
    plan->fk = d_fk;

    gridder_plan->dirty_image = dirty_image; //
    checkCudaError(cudaMemcpy(d_fk, dirty_image, sizeof(CUCPX) * nxdirty * nydirty, cudaMemcpyHostToDevice));

    // how to use weight flag and frequency
    for (int i = 0; i < nchan; i++)
    {
        // pre_setting
        // 1. u, v, w * f_over_c
        // 2. *pixelsize(*2pi)
        // 3. * rescale ratio
        // pre_setting(d_u, d_v, d_w, d_vis, plan, gridder_plan);
        // memory transfer (vis belong to this channel and weight)
        // checkCudaError(cudaMemcpy(d_vis, vis, nrow * sizeof(CUCPX), cudaMemcpyHostToDevice)); //
        // shift to corresponding range
        ier = gridder_execution(plan, gridder_plan);
        if (ier == 1)
        {
            printf("errors in gridder execution\n");
            return ier;
        }
        checkCudaError(cudaMemcpy(gridder_plan->kv.vis, plan->d_c, sizeof(CUCPX) * nrow,
                                  cudaMemcpyDeviceToHost));
    }
    printf("exection finished\n");
#ifdef PRINT
    printf("result printing...\n");
    for (int i = 0; i < nxdirty; i++)
    {
        for (int j = 0; j < nydirty; j++)
        {
            printf("%.5lf ", gridder_plan->dirty_image[i * nydirty + j].real());
        }
        printf("\n");
    }
#endif
    PCS pi_ratio = 1;
    if (!gridder_plan->kv.pirange)
        pi_ratio = 2 * PI;
    int print_row = nrow;
    if (nrow >= 1e4)
    {
        print_row = 10;
    }
    PCS *truth = (PCS *)malloc(sizeof(PCS) * nrow);
    for (int k = 0; k < print_row; k++)
    {
        CPX temp(0.0, 0.0);
        for (int i = 0; i < nxdirty; i++)
        {
            for (int j = 0; j < nydirty; j++)
            {
                PCS n_lm = sqrt(1.0 - pow(gridder_plan->pixelsize_x * (i - nxdirty / 2), 2) - pow(gridder_plan->pixelsize_y * (j - nydirty / 2), 2));
                PCS phase = -f0 / SPEEDOFLIGHT * (u[k] * pi_ratio * gridder_plan->pixelsize_x * (i - nxdirty / 2) + v[k] * pi_ratio * gridder_plan->pixelsize_y * (j - nydirty / 2) - w[k] * pi_ratio * (n_lm - 1));
                temp += dirty_image[i * nydirty + j] / n_lm * exp(phase * IMA);
            }
        }
        truth[k] = temp.real();
    }

    printf("portion of result and ground truth printing...\n");
    for (int i = 0; i < 10; i++)
    {
        printf("(%lf,%lf)", gridder_plan->kv.vis[i].real(), truth[i]);
    }
    printf("\n");
    double max = 0;
    double l2_max = 0;
    double sum_fk = 0;

    for (int i = 0; i < print_row; i++)
    {
        double temp = abs(truth[i] - gridder_plan->kv.vis[i].real());
        if (temp > max)
        {
            max = temp;
        }
        l2_max += temp;
        sum_fk += abs(gridder_plan->kv.vis[i].real());
    }
    printf("maximal abs error %.3g, l2 error %.3g\n", max, l2_max / sum_fk);
    free(truth);
    printf("---------------------------------------------------------------------------------------------------\n");
    plan->dim = 3;
    ier = gridder_destroy(plan, gridder_plan);
    if (ier == 1)
    {
        printf("errors in gridder destroy\n");
        return ier;
    }

    return ier;
}