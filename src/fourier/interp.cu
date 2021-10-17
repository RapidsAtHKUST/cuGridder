
/*
    Interp on GPU
	Fucntions:
		1. val_kernel_vec
		2. interp_*d_nputsdriven
		
*/

#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <helper_cuda.h>
//#include <thrust/extrema.h>
#include "interp.h"

static __inline__ __device__ void val_kernel_vec(PCS *ker, const PCS x, const double kw, const double es_c,
												 const double es_beta)
{
	//get vector of kernel function values
	for (int i = 0; i < kw; i++)
	{
		ker[i] = (abs(x + i) >= kw / 2.0) ? 0.0 : exp(es_beta * (sqrt(1.0 - es_c * (x + i) * (x + i))));
	}
}

__global__ void interp_1d_nputsdriven(PCS *x, CUCPX *c, CUCPX *fw, int M,
									const int ns, int nf1, PCS es_c, PCS es_beta, int pirange)
{
	/*
	Output driven convolution
		x - output location, range: [-pi,pi)
		c - complex number result
		fw - input
		M - number of nupts
		ns - kernel width
		nf1 - upts after upsampling
		es_ - gridding kernel related factors
		pirange - in pi range or not
	*/

	int xstart, xend; // first grid point for this coordinate
	int ix;
	int indx;
	PCS ker1[MAX_KERNEL_WIDTH]; // values of kernel function evaluation

	PCS temp1;
	int idx;

	for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x)
	{
		//value of x, shift and rescale to [0,N) and get the locations
		temp1 = SHIFT_RESCALE(x[idx], nf1, pirange);
		
		xstart = ceil(temp1 - ns / 2.0);
		xend = floor(temp1 + ns / 2.0);
		
		PCS x_1 = (PCS)xstart - temp1; // distance from first in range grid point to input coordinate
		val_kernel_vec(ker1, x_1, ns, es_c, es_beta);
		for (int xx = xstart; xx <= xend; xx++)
		{
			ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
			indx = ix;
			PCS kervalue = ker1[xx - xstart];
			c[idx].x += fw[indx].x * kervalue;
			c[idx].y += fw[indx].y * kervalue;
		}
		
	}
}

// 2D for w-stacking. 1D + 2D for improved WS will consume more memory
__global__ void interp_2d_nputsdriven(PCS *x, PCS *y, CUCPX *c, CUCPX *fw, int M,
									const int ns, int nf1, int nf2, PCS es_c, PCS es_beta, int pirange)
{
	/*
		x, y - range [-pi,pi)
		c - complex number
		fw - result
		M - number of nupts
		ns - kernel width
		nf1, nf2 - upts
		es_ - gridding kernel related factors
		pirange - 1
	*/
	//need to revise
	int xstart, ystart, xend, yend;
	int ix, iy;
	int indx;
	PCS ker1[MAX_KERNEL_WIDTH];
	PCS ker2[MAX_KERNEL_WIDTH];

	PCS temp1, temp2;
	int idx;
	

	for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x)
	{

		//value of x, shift and rescale to [0,N) and get the locations
		temp1 = SHIFT_RESCALE(x[idx], nf1, pirange);
		temp2 = SHIFT_RESCALE(y[idx], nf2, pirange);
		// temp1 = RESCALE(x[idx],nf1,pirange);
		// temp2 = RESCALE(y[idx],nf1,pirange);

		xstart = ceil(temp1 - ns / 2.0);
		ystart = ceil(temp2 - ns / 2.0);
		xend = floor(temp1 + ns / 2.0);
		yend = floor(temp2 + ns / 2.0);

		PCS x_1 = (PCS)xstart - temp1; //cell
		PCS y_1 = (PCS)ystart - temp2;
		val_kernel_vec(ker1, x_1, ns, es_c, es_beta);
		val_kernel_vec(ker2, y_1, ns, es_c, es_beta);
		for (int yy = ystart; yy <= yend; yy++)
		{
			temp1 = ker2[yy - ystart];
			for (int xx = xstart; xx <= xend; xx++)
			{
				ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
				iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
				indx = ix + iy * nf1;
				temp2 = ker1[xx - xstart];
				PCS kervalue = temp1 * temp2;
				c[idx].x += fw[indx].x * kervalue;
			    c[idx].y += fw[indx].y * kervalue;
			}
		}
	}
}

__global__ void interp_3d_nputsdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int M,
									const int ns, int nf1, int nf2, int nf3, PCS es_c, PCS es_beta, int pirange)
{
	/*
		x, y, z - range [-pi,pi)
		c - complex number
		fw - result
		M - number of nupts
		ns - kernel width
		nf1, nf2, nf3 - upts
		es_ - gridding kernel related factors
		pirange - 1
	*/

	int idx;
	idx = blockDim.x * blockIdx.x + threadIdx.x;
	int xx, yy, zz, ix, iy, iz;
	int indx;

	PCS ker1[MAX_KERNEL_WIDTH];
	PCS ker2[MAX_KERNEL_WIDTH];
	PCS ker3[MAX_KERNEL_WIDTH];

	PCS temp1, temp2, temp3;

	assert(pirange == 1); // check, the x y z should be in range [-pi,pi)

	for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < M; idx += blockDim.x * gridDim.x)
	{

		//value of x, shift and rescale to [0,N) and get the locations
		temp1 = SHIFT_RESCALE(x[idx], nf1, pirange);
		temp2 = SHIFT_RESCALE(y[idx], nf2, pirange);
		temp3 = SHIFT_RESCALE(z[idx], nf3, pirange);


		int xstart = ceil(temp1 - ns / 2.0);
		int ystart = ceil(temp2 - ns / 2.0);
		int zstart = ceil(temp3 - ns / 2.0);
		int xend = floor(temp1 + ns / 2.0);
		int yend = floor(temp2 + ns / 2.0);
		int zend = floor(temp3 + ns / 2.0);

		PCS x1 = (PCS)xstart - temp1;
		PCS y1 = (PCS)ystart - temp2;
		PCS z1 = (PCS)zstart - temp3;

		val_kernel_vec(ker1, x1, ns, es_c, es_beta);
		val_kernel_vec(ker2, y1, ns, es_c, es_beta);
		val_kernel_vec(ker3, z1, ns, es_c, es_beta);

		for (zz = zstart; zz <= zend; zz++)
		{
			temp3 = ker3[zz - zstart];
			for (yy = ystart; yy <= yend; yy++)
			{
				temp2 = ker2[yy - ystart];
				for (xx = xstart; xx <= xend; xx++)
				{
					//due to the peroid, the index out of range need to be handle
					ix = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
					iy = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
					iz = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
					indx = ix + iy * nf1 + iz * nf1 * nf2;

					temp1 = ker1[xx - xstart];
					PCS kervalue = temp1 * temp2 * temp3;
					c[idx].x += fw[indx].x * kervalue;
			        c[idx].y += fw[indx].y * kervalue;
					//printf("the out id %d kervalue %2.2g\n",outidx,kervalue);
				}
			}
			
		}
		//if((idx/blockDim.x+1)*blockDim.x<M){ __syncthreads(); }
	}
}