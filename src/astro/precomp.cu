/*
Some precomputation related radio astronomy
    Sampling
    Vis * weight
    ...
*/
#include "datatype.h"
#include "curafft_plan.h"
#include "ragridder_plan.h"
#include "deconv.h"
#include "utils.h"
#include "precomp.h"

__global__ void get_effective_coordinate(PCS *u, PCS *v, PCS *w, PCS f_over_c, int pirange, int nrow)
{
    /*
        u, v, w - coordinate
        f_over_c - frequency divide speed of light
        pirange - 1 in [-pi,pi), 0 - [-0.5,0.5)
        nrow - number of coordinates
    */

    int idx;
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < nrow; idx += gridDim.x * blockDim.x)
    {
        //if(idx==0) printf("Before scaling w %lf, f over c %lf\n",w[idx],f_over_c);
        u[idx] *= f_over_c;
        v[idx] *= f_over_c;
        w[idx] *= f_over_c;
        if (!pirange)
        {
            u[idx] *= 2 * PI;
            v[idx] *= 2 * PI;
            w[idx] *= 2 * PI;
        }
        // if(idx==0) printf("After scaling w %lf, f over c %lf\n",w[idx],f_over_c);
    }
}

void get_effective_coordinate_invoker(PCS *d_u, PCS *d_v, PCS *d_w, PCS f_over_c, int pirange, int nrow)
{
    int blocksize = 512;
    // printf("nrow %d, foc %lf",nrow,f_over_c);
    get_effective_coordinate<<<(nrow - 1) / blocksize + 1, blocksize>>>(d_u, d_v, d_w, f_over_c, pirange, nrow);
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void get_effective_coordinate(PCS *u, PCS *v, PCS *w, PCS f_over_c, int pirange, int nrow, int sign)
{
    /*
        u, v, w - coordinate
        f_over_c - frequency divide speed of light
        pirange - 1 in [-pi,pi), 0 - [-0.5,0.5)
        nrow - number of coordinates
    */

    int idx;
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < nrow; idx += gridDim.x * blockDim.x)
    {
        //if(idx==0) printf("Before scaling w %lf, f over c %lf\n",w[idx],f_over_c);
        u[idx] *= f_over_c;
        v[idx] *= f_over_c;
        w[idx] *= f_over_c * sign;
        if (!pirange)
        {
            u[idx] *= 2 * PI;
            v[idx] *= 2 * PI;
            w[idx] *= 2 * PI;
        }
        // if(idx==0) printf("After scaling w %lf, f over c %lf\n",w[idx],f_over_c);
    }
}

void get_effective_coordinate_invoker(PCS *d_u, PCS *d_v, PCS *d_w, PCS f_over_c, int pirange, int nrow, int sign)
{
    int blocksize = 512;
    // printf("nrow %d, foc %lf",nrow,f_over_c);
    get_effective_coordinate<<<(nrow - 1) / blocksize + 1, blocksize>>>(d_u, d_v, d_w, f_over_c, pirange, nrow, sign);
    checkCudaErrors(cudaDeviceSynchronize());
}
__global__ void gridder_rescaling_real(PCS *x, PCS scale_ratio, int N)
{
    int idx;
    for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        x[idx] *= scale_ratio;
    }
}

void pre_setting(PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_vis, CURAFFT_PLAN *plan, ragridder_plan *gridder_plan)
{
    PCS f_over_c = gridder_plan->kv.frequency[gridder_plan->cur_channel] / SPEEDOFLIGHT;
    PCS xpixelsize = gridder_plan->pixelsize_x;
    PCS ypixelsize = gridder_plan->pixelsize_y;
    int pirange = gridder_plan->kv.pirange;
    int nrow = gridder_plan->nrow;
    int N = nrow;
    int blocksize = 512;
    //----------w max min num_w w_0--------------
    gridder_plan->w_max *= f_over_c;
    gridder_plan->w_min *= f_over_c;
    PCS w_0 = gridder_plan->w_min - gridder_plan->delta_w * (plan->copts.kw - 1); // first plane
    gridder_plan->w_0 = w_0;
    int num_w = ((gridder_plan->w_max - gridder_plan->w_min) / gridder_plan->delta_w + plan->copts.kw) + 4;

    num_w = num_w * plan->copts.upsampfac;
    gridder_plan->num_w = num_w;

    if (gridder_plan->cur_channel != 0)
    {
        gridder_plan->w_max /= gridder_plan->kv.frequency[gridder_plan->cur_channel - 1] / SPEEDOFLIGHT;
        gridder_plan->w_min /= gridder_plan->kv.frequency[gridder_plan->cur_channel - 1] / SPEEDOFLIGHT;
    }
    // printf("frequency over  speed of light %lf\n",f_over_c);
    // printf("scaling method: w_max %lf, w_min %lf\n",gridder_plan->w_max, gridder_plan->w_min);
    PCS previous_sr = gridder_plan->w_s_r;
    gridder_plan->w_s_r = std::max(abs(gridder_plan->w_max), abs(gridder_plan->w_min)) / PI;
    //----------plan reset--------------
    plan->nf3 = num_w;
    //plan->batchsize = min(4,num_w);
    plan->batchsize = num_w; // not support for particle calculation
    int N1 = plan->ms;
    int N2 = plan->mt;

    if (gridder_plan->w_term_method)
    {
        // improved_ws
        checkCudaErrors(cudaFree(plan->fwkerhalf3)); //if cur channel = 0
        checkCudaErrors(cudaMalloc((void **)&plan->fwkerhalf3, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1)));
        PCS *d_k;
        checkCudaErrors(cudaMalloc((void **)&d_k, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1)));
        w_term_k_generation(d_k, plan->ms, plan->mt, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y); //some issues
#ifdef DEBUG
        printf("k printing...\n");
        PCS *h_k;
        h_k = (PCS *)malloc(sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1));
        cudaMemcpy(h_k, d_k, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N2 / 2 + 1; i++)
        {
            for (int j = 0; j < N1 / 2 + 1; j++)
            {
                /* code */
                printf("%.6lf ", h_k[i * (N1 / 2 + 1) + j]);
            }
            printf("\n");
        }
#endif
        fourier_series_appro_invoker(plan->fwkerhalf3, d_k, plan->copts, (N1 / 2 + 1) * (N2 / 2 + 1), num_w / 2 + 1); // correction with k, cautious
        checkCudaErrors(cudaFree(d_k));
        //
#ifdef DEBUG
        PCS *fwkerhalf3 = (PCS *)malloc(sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1));
        cudaMemcpy(fwkerhalf3, plan->fwkerhalf3, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N2 / 2 + 1; i++)
        {
            for (int j = 0; j < N1 / 2 + 1; j++)
            {
                printf("%.lf ", fwkerhalf3[i * (N1 / 2 + 1) + j]);
            }
            printf("\n");
        }
#endif
        if (plan->fw != NULL)
        {
            checkCudaErrors(cudaFree(plan->fw));
            plan->fw = NULL;
        }

        checkCudaErrors(cudaMalloc((void **)&plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3));
        checkCudaErrors(cudaMemset(plan->fw, 0, plan->nf3 * plan->nf1 * plan->nf2 * sizeof(CUCPX)));
    }

    int n[] = {plan->nf2, plan->nf1};
    int inembed[] = {plan->nf2, plan->nf1};
    int onembed[] = {plan->nf2, plan->nf1};
    cufftPlanMany(&plan->fftplan, 2, n, inembed, 1, inembed[0] * inembed[1],
                  onembed, 1, onembed[0] * onembed[1], CUFFT_TYPE, plan->nf3);
    // ---------get effective coordinates---------
    get_effective_coordinate<<<(N - 1) / blocksize + 1, blocksize>>>(d_u, d_v, d_w, f_over_c, pirange, nrow);
    checkCudaErrors(cudaDeviceSynchronize());

    // get_max_min(gridder_plan->w_max,gridder_plan->w_min,plan->d_w,gridder_plan->nrow);
    // printf("scaling method: w_max %lf, w_min %lf\n",gridder_plan->w_max, gridder_plan->w_min);

    checkCudaErrors(cudaDeviceSynchronize());
    gridder_plan->kv.pirange = 1;
    plan->copts.pirange = 1;
    // ----------------rescaling-----------------
    PCS scaling_ratio = xpixelsize;
    gridder_rescaling_real<<<(N - 1) / blocksize + 1, blocksize>>>(d_u, scaling_ratio, nrow);
    checkCudaErrors(cudaDeviceSynchronize());
    scaling_ratio = ypixelsize;
    gridder_rescaling_real<<<(N - 1) / blocksize + 1, blocksize>>>(d_v, scaling_ratio, nrow);
    checkCudaErrors(cudaDeviceSynchronize());
    gridder_rescaling_real<<<(N - 1) / blocksize + 1, blocksize>>>(d_w, previous_sr / gridder_plan->w_s_r, nrow); // scaling w to [-pi, pi)
    checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG
    PCS *w = (PCS *)malloc(sizeof(PCS) * nrow);
    checkCudaErrors(cudaMemcpy(w, d_w, sizeof(PCS) * nrow, cudaMemcpyDeviceToHost));
    printf("effective coordinate printing...\n");
    for (int i = 0; i < nrow; i++)
    {
        printf("%.3lf ", w[i]);
    }
#endif
    // ------------vis * flag * weight--------+++++
    // memory transfer (vis belong to this channel and weight)
    checkCudaErrors(cudaMemcpy(d_vis, gridder_plan->kv.vis + nrow * gridder_plan->cur_channel, nrow * sizeof(CUCPX), cudaMemcpyHostToDevice)); //
}

__global__ void k_generation(PCS *k, int N1, int N2, PCS xpixelsize, PCS ypixelsize)
{
    int idx;
    int N = (N1 / 2 + 1) * (N2 / 2 + 1);
    for (idx = threadIdx.x + blockDim.x * blockIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        int row = idx / (N1 / 2 + 1);
        int col = idx % (N1 / 2 + 1);
        k[idx] = (sqrt(1.0 - pow(row * xpixelsize, 2) - pow(col * ypixelsize, 2)) - 1);
    }
}

int w_term_k_generation(PCS *k, int N1, int N2, PCS xpixelsize, PCS ypixelsize)
{
    /*
        W term k array generation
        Output:
            k - size: [nf1/2+1, nf2/2+1] (due to the parity, just calculate 1/4 of the result)
            The value of k[i] is set based on z = sqrt(1 - l^2 - m^2) - 1
        
    */
    // k array generation

    int N = (N1 / 2 + 1) * (N2 / 2 + 1);
    int blocksize = 512;
    k_generation<<<(N - 1) / blocksize + 1, blocksize>>>(k, N1, N2, xpixelsize, ypixelsize);
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}

__global__ void explicit_gridder(int N1, int N2, int nrow, PCS *u, PCS *v, PCS *w, CUCPX *vis,
                                 CUCPX *dirty, PCS f_over_c, PCS row_pix_size, PCS col_pix_size, int pirange)
{
    /*
        N1,N2 - width, height 
        row_pix_size, col_pix_size - xpixsize, ypixsize
    */
    int idx;
    int row;
    int col;
    PCS l, m, n_lm;
    CUCPX temp;
    for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N1 * N2; idx += gridDim.x * blockDim.x)
    {
        temp.x = 0.0;
        temp.y = 0.0;
        row = idx / N1 - N2 / 2;
        col = idx % N1 - N1 / 2;
        l = row * row_pix_size;
        m = col * col_pix_size;
        n_lm = sqrt(1.0 - pow(l, 2) - pow(m, 2));
        for (int i = 0; i < nrow; i++)
        {
            PCS phase = f_over_c * (l * u[i] + m * v[i] + (n_lm - 1) * w[i]);
            if (pirange != 1)
                phase = phase * 2 * PI;
            temp.x += vis[i].x * cos(phase) - vis[i].y * sin(phase);
            temp.y += vis[i].x * sin(phase) + vis[i].y * cos(phase);
        }

        dirty[idx].x = temp.x / n_lm; // add values of all channels
        dirty[idx].y = temp.y / n_lm;
    }
}

int explicit_gridder_invoker(ragridder_plan *gridder_plan, PCS e)
{
    int ier = (int) e;
    int nchan = gridder_plan->channel;
    int nrow = gridder_plan->nrow;
    int N1 = gridder_plan->width;
    int N2 = gridder_plan->height;
    int pirange = gridder_plan->kv.pirange;
    PCS xpixsize = gridder_plan->pixelsize_x;
    PCS ypixsize = gridder_plan->pixelsize_y;
    PCS *d_u, *d_v, *d_w;
    CUCPX *d_vis, *d_dirty;
    checkCudaErrors(cudaMalloc((void **)&d_u, sizeof(PCS) * nrow));
    checkCudaErrors(cudaMalloc((void **)&d_v, sizeof(PCS) * nrow));
    checkCudaErrors(cudaMalloc((void **)&d_w, sizeof(PCS) * nrow));
    checkCudaErrors(cudaMalloc((void **)&d_vis, sizeof(CUCPX) * nrow));
    checkCudaErrors(cudaMalloc((void **)&d_dirty, sizeof(CUCPX) * N1 * N2));
    checkCudaErrors(cudaMemset(d_dirty, 0, sizeof(CUCPX) * N1 * N2));

    checkCudaErrors(cudaMemcpy(d_u, gridder_plan->kv.u, sizeof(PCS) * nrow, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_v, gridder_plan->kv.v, sizeof(PCS) * nrow, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_w, gridder_plan->kv.w, sizeof(PCS) * nrow, cudaMemcpyHostToDevice));

    int blocksize = 512;
    PCS f_over_c;
    for (int i = 0; i < nchan; i++)
    {
        checkCudaErrors(cudaMemcpy(d_vis, gridder_plan->kv.vis + i * nrow, sizeof(CUCPX) * nrow, cudaMemcpyHostToDevice));
        f_over_c = gridder_plan->kv.frequency[i] / SPEEDOFLIGHT;
        explicit_gridder<<<(N1 * N2 - 1) / blocksize + 1, blocksize>>>(N1, N2, nrow, d_u, d_v, d_w, d_vis,
                                                                       d_dirty, f_over_c, xpixsize, ypixsize, pirange); // blocksize can not be 1024
        checkCudaErrors(cudaDeviceSynchronize());
    }
    checkCudaErrors(cudaMemcpy(gridder_plan->dirty_image, d_dirty, sizeof(CUCPX) * N1 * N2, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_u));
    checkCudaErrors(cudaFree(d_v));
    checkCudaErrors(cudaFree(d_w));
    checkCudaErrors(cudaFree(d_vis));
    checkCudaErrors(cudaFree(d_dirty));
    return ier;
}