//------ convolutional gridding -------
/*
    Gridding on GPU
	Fucntions:
		1. val_kernel_vec
		2. conv_*d_nputsdriven
		3. partial_3d_conv_sorted
*/

#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <helper_cuda.h>
//#include <thrust/extrema.h>
#include "conv.h"
#define SHARED_SIZE_3D_1 512

static __inline__ __device__ void kervalue_evaluate(PCS &ker, const PCS x, const double kw, const double es_c,
												 const double es_beta)
{	
	ker = (abs(x) >= kw / 2.0) ? 0.0 : exp(es_beta * (sqrt(1.0 - es_c * x  * x )));
}



static __inline__ __device__ void val_kernel_vec(PCS *ker, const PCS x, const double kw, const double es_c,
												 const double es_beta)
{
	//get vector of kernel function values
	for (int i = 0; i < kw; i++)
	{
		ker[i] = (abs(x + i) >= kw / 2.0) ? 0.0 : exp(es_beta * (sqrt(1.0 - es_c * (x + i) * (x + i))));
	}
}

__global__ void counting_hive(int *hive_count, int *histo_count, unsigned long int M, int hivesize){
	unsigned int idx;
	for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x){
		hive_count[idx] = histo_count[idx*hivesize];
		// printf("%d, %d\n",idx, hive_count[idx]);
	}
}

__global__ void conv_1d_nputsdriven(PCS *x, CUCPX *c, CUCPX *fw, int M,
									const int ns, int nf1, PCS es_c, PCS es_beta, int pirange)
{
	/*
	Input driven convolution
		x - input location, range: [-pi,pi)
		c - complex number
		fw - result
		M - number of nupts
		ns - kernel width
		nf1 - upts after upsampling
		es_ - gridding kernel related factors
		pirange - in pi range or not
	*/

	int xstart, xend; // first grid point for this coordinate
	int ix;
	int outidx;
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
			outidx = ix;
			PCS kervalue = ker1[xx - xstart];
			atomicAdd(&fw[outidx].x, c[idx].x * kervalue); //avoid concurrent write
			atomicAdd(&fw[outidx].y, c[idx].y * kervalue);
		}
	}
}

// 2D for w-stacking. 1D + 2D for improved WS will consume more memory
__global__ void conv_2d_nputsdriven(PCS *x, PCS *y, CUCPX *c, CUCPX *fw, int M,
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
	int outidx;
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
				outidx = ix + iy * nf1;
				temp2 = ker1[xx - xstart];
				PCS kervalue = temp1 * temp2;
				atomicAdd(&fw[outidx].x, c[idx].x * kervalue);
				atomicAdd(&fw[outidx].y, c[idx].y * kervalue);
			}
		}
	}
}

__global__ void conv_3d_nputsdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int M,
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
	int outidx;

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
					outidx = ix + iy * nf1 + iz * nf1 * nf2;

					temp1 = ker1[xx - xstart];
					PCS kervalue = temp1 * temp2 * temp3;
					// fw[outidx].x += c[idx].x * kervalue;
					// fw[outidx].y += c[idx].y * kervalue;
					// if(outidx==616)printf("%lf,%lf,%lf,%lf\n",x[idx],x1+xx-xstart,y1+yy-ystart,z1+zz-zstart);
					atomicAdd(&fw[outidx].x, c[idx].x * kervalue);
					atomicAdd(&fw[outidx].y, c[idx].y * kervalue);
					//printf("the out id %d kervalue %2.2g\n",outidx,kervalue);
				}
			}
		}
		//if((idx/blockDim.x+1)*blockDim.x<M){ __syncthreads(); }
	}
}

__global__ void conv_3d_outputdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange){
	/*
		blocksize = 8*8*8 if change may need to revise
		another method also load intput into shared memroy by multi times

		remove some variable or put to constant memory remove nbin
	*/
	
	unsigned long int idx; // one hive by one hive
	unsigned long int M = nbin_x; // the threads are padded
	M *= nbin_y;
	M *= nbin_z;
	
	for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < M; idx += blockDim.x * gridDim.x)
	{
		int hive_x, hive_y, hive_z;
		unsigned long int outidx;
		// int bin_idx;
		// load to shared memory __synchronize
		// extern __shared__ CUCPX sh_fw[];
		
		int cur_hive_idx = blockIdx.x; // current hive idx
		hive_x = cur_hive_idx % nhive_x;
		hive_y = cur_hive_idx / nhive_x % nhive_y;
		hive_z = cur_hive_idx / (nhive_x*nhive_y);

		// bin_idx = threadIdx.x % (nbin_x / hive_x) + threadIdx.x / (nbin_x / hive_x) % (nbin_y / hive_y) + threadIdx.x;
		// idx in hive + hive_x * hivesize_x
		int bin_x = threadIdx.x % (nbin_x / nhive_x) + hive_x * (nbin_x / nhive_x);
		int bin_y = threadIdx.x / (nbin_x / nhive_x) % (nbin_y / nhive_y) + hive_y * (nbin_y / nhive_y);
		int bin_z = threadIdx.x / ((nbin_x / nhive_x) * (nbin_y / nhive_y)) + hive_z * (nbin_z / nhive_z);
		outidx = nf1*nf2;
		outidx *= bin_z;
		outidx += bin_x + bin_y * nf1;
		
		int flag = 1;
		int cur_hive_x;
		int cur_hive_y;
		int cur_hive_z; 
		// start_hive_idx[1] = cur_hive_idx - nhive_x - 1;
		// start_hive_idx[2] = start_hive_idx[1] + nhive_x*nhive_y;
		// start_hive_idx[0] = start_hive_idx[1] - nhive_x*nhive_y; 
		
		
		if(bin_x<nf1&&bin_y<nf2&&bin_z<nf3){
			for(int i = 0; i<3; i++){
				for(int j=0; j<9; j++){
					flag = 1;
					
					cur_hive_x = hive_x + j % 3 - 1;
					cur_hive_y = hive_y + j / 3 - 1;
					cur_hive_z = hive_z + i - 1;

					
					if(cur_hive_x >= nhive_x || cur_hive_x < 0) nhive_x<3? flag=0: cur_hive_x -= ((cur_hive_x > 0) - (cur_hive_x < 0))*nhive_x;
					if(cur_hive_y >= nhive_y || cur_hive_y < 0) nhive_y<3? flag=0: cur_hive_y -= ((cur_hive_y > 0) - (cur_hive_y < 0))*nhive_y;
					if(cur_hive_z >= nhive_z || cur_hive_z < 0) nhive_z<3? flag=0: cur_hive_z -= ((cur_hive_z > 0) - (cur_hive_z < 0))*nhive_z;
					// if(outidx==616)printf("%d,%d,%d,%d\n",cur_hive_idx,cur_hive_x,cur_hive_y,cur_hive_z);
					if (flag==0) continue; // exceeding the boundart and nf < 3
					cur_hive_idx = cur_hive_x + cur_hive_y * nhive_x + cur_hive_z * nhive_x * nhive_y;
					// if(outidx==616)printf("%d,%d,%d,%d,%lu\n",cur_hive_idx,cur_hive_x,cur_hive_y,cur_hive_z,idx);
					
					

					//if(cur_hive_idx>=nhive_x*nhive_y*nhive_z||cur_hive_idx<0)printf("%d,%d,%d,%d,%d ",cur_hive_idx, hive_x,hive_y, hive_z,flag);
					for(int k=hive_count[cur_hive_idx]; k<hive_count[cur_hive_idx+1]; k++){ // here problem
						// kernel evaluation
						PCS ker;
						PCS kervalue = 1.0;

						PCS temp = SHIFT_RESCALE(x[k], nf1, pirange); //save
						temp = abs(temp-bin_x);
						//++++ break if not in range
						if(temp>nf1/2.0)temp = nf1 - temp;
						if(abs(temp)>ns/2.0)continue; 

						PCS temp2 = SHIFT_RESCALE(y[k], nf2, pirange);
						temp2 = abs(temp2-bin_y);
						if(temp2>nf2/2.0)temp2 = nf2 - temp2;
						if(abs(temp2)>ns/2.0)continue;

						PCS temp3 = SHIFT_RESCALE(z[k], nf3, pirange);
						temp3 = abs(temp3-bin_z);
						if(temp3>nf3/2.0)temp3 = nf3 - temp3;
						if(abs(temp3)>ns/2.0)continue;
						// if(outidx==0)printf("%lf,%lf,%lf,%lf\n",temp,temp2,temp3,c[k].x);

						kervalue_evaluate(ker, temp, ns, es_c, es_beta);
						kervalue = kervalue * ker;

						kervalue_evaluate(ker, temp2, ns, es_c, es_beta);
						kervalue = kervalue * ker;

						kervalue_evaluate(ker, temp3, ns, es_c, es_beta);
						kervalue = kervalue * ker;
						// if(outidx==616)printf("%lf,%lu,%d,%d,%d\n",x[k],idx,cur_hive_x,cur_hive_y,cur_hive_z);
						
						// if(outidx==nf1*nf2-1)printf("%lf,%lf,%lf\n",x[k],temp,kervalue);
						fw[outidx].x += c[k].x * kervalue;
						fw[outidx].y += c[k].y * kervalue;
						// if(outidx==nf1*nf2*nf3-10)printf("%lf\n",kervalue);

					}
				}
			}
		}
	}
}



__global__ void conv_3d_outputdriven_shared(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange){
	/*
		blocksize = 8*8*8 if change may need to revise
		another method also load intput into shared memroy by multi times, currently just load output to sharedmem

		remove some variable or put to constant memory remove nbin
	*/
	
	unsigned long int idx; // one hive by one hive
	unsigned long int M = nbin_x; // the threads are padded
	M *= nbin_y;
	M *= nbin_z;
	__shared__ CUCPX sh_fw[SHARED_SIZE_3D_1];
	for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < M; idx += blockDim.x * gridDim.x)
	{
		int hive_x, hive_y, hive_z;
		unsigned long int outidx;
		// int bin_idx;
		// load to shared memory __synchronize
		
		
		
		sh_fw[threadIdx.x].x = 0;
		sh_fw[threadIdx.x].y = 0;
		//__syncthreads();
		
		int cur_hive_idx = blockIdx.x; // current hive idx
		hive_x = cur_hive_idx % nhive_x;
		hive_y = cur_hive_idx / nhive_x % nhive_y;
		hive_z = cur_hive_idx / (nhive_x*nhive_y);

		// bin_idx = threadIdx.x % (nbin_x / hive_x) + threadIdx.x / (nbin_x / hive_x) % (nbin_y / hive_y) + threadIdx.x;
		// idx in hive + hive_x * hivesize_x
		int bin_x = threadIdx.x % (nbin_x / nhive_x) + hive_x * (nbin_x / nhive_x);
		int bin_y = threadIdx.x / (nbin_x / nhive_x) % (nbin_y / nhive_y) + hive_y * (nbin_y / nhive_y);
		int bin_z = threadIdx.x / ((nbin_x / nhive_x) * (nbin_y / nhive_y)) + hive_z * (nbin_z / nhive_z);
		outidx = nf1*nf2;
		outidx *= bin_z;
		outidx += bin_x + bin_y * nf1;
		// if(idx==1144)printf("idx %d, %d, %d, %d\n",cur_hive_idx,(threadIdx.x / ((nbin_x / nhive_x) * (nbin_y / hive_y))) ,(nbin_x / nhive_x) * (nbin_y / hive_y),threadIdx.x);
		// if(idx==5)printf("the outidx is %lu, %d,%d,%d,%d\n",outidx,hive_x,hive_y,hive_z ,threadIdx.x);
		int flag = 1;
		int cur_hive_x;
		int cur_hive_y;
		int cur_hive_z; 
		// start_hive_idx[1] = cur_hive_idx - nhive_x - 1;
		// start_hive_idx[2] = start_hive_idx[1] + nhive_x*nhive_y;
		// start_hive_idx[0] = start_hive_idx[1] - nhive_x*nhive_y; 
		
		
		if(bin_x<nf1&&bin_y<nf2&&bin_z<nf3){
			for(int i = 0; i<3; i++){
				for(int j=0; j<9; j++){
					flag = 1;
					
					cur_hive_x = hive_x + j % 3 - 1;
					cur_hive_y = hive_y + j / 3 - 1;
					cur_hive_z = hive_z + i - 1;

					// cur_hive_idx = start_hive_idx[i] + j/3*nhive_x + j%3;
					// hive_x = cur_hive_idx % nhive_x;
					// hive_y = cur_hive_idx / nhive_x % nhive_y;
					// hive_z = cur_hive_idx / (nhive_x * nhive_y);
					// handling the exceeding boundary issue
					
					if(cur_hive_x >= nhive_x || cur_hive_x < 0) nhive_x<3? flag=0: cur_hive_x -= ((cur_hive_x > 0) - (cur_hive_x < 0))*nhive_x;
					if(cur_hive_y >= nhive_y || cur_hive_y < 0) nhive_y<3? flag=0: cur_hive_y -= ((cur_hive_y > 0) - (cur_hive_y < 0))*nhive_y;
					if(cur_hive_z >= nhive_z || cur_hive_z < 0) nhive_z<3? flag=0: cur_hive_z -= ((cur_hive_z > 0) - (cur_hive_z < 0))*nhive_z;
					// if(outidx==616)printf("%d,%d,%d,%d\n",cur_hive_idx,cur_hive_x,cur_hive_y,cur_hive_z);
					if (flag==0) continue; // exceeding the boundart and nf < 3
					cur_hive_idx = cur_hive_x + cur_hive_y * nhive_x + cur_hive_z * nhive_x * nhive_y;
					// if(outidx==616)printf("%d,%d,%d,%d,%lu\n",cur_hive_idx,cur_hive_x,cur_hive_y,cur_hive_z,idx);
					
					

					//if(cur_hive_idx>=nhive_x*nhive_y*nhive_z||cur_hive_idx<0)printf("%d,%d,%d,%d,%d ",cur_hive_idx, hive_x,hive_y, hive_z,flag);
					for(int k=hive_count[cur_hive_idx]; k<hive_count[cur_hive_idx+1]; k++){ // here problem
						// kernel evaluation
						PCS ker;
						PCS kervalue = 1.0;

						PCS temp = SHIFT_RESCALE(x[k], nf1, pirange); //save
						temp = abs(temp-bin_x);
						//++++ break if not in range
						if(temp>nf1/2.0)temp = nf1 - temp;
						if(abs(temp)>ns/2.0)continue; 

						PCS temp2 = SHIFT_RESCALE(y[k], nf2, pirange);
						temp2 = abs(temp2-bin_y);
						if(temp2>nf2/2.0)temp2 = nf2 - temp2;
						if(abs(temp2)>ns/2.0)continue;

						PCS temp3 = SHIFT_RESCALE(z[k], nf3, pirange);
						temp3 = abs(temp3-bin_z);
						if(temp3>nf3/2.0)temp3 = nf3 - temp3;
						if(abs(temp3)>ns/2.0)continue;

						kervalue_evaluate(ker, temp, ns, es_c, es_beta);
						kervalue = kervalue * ker;

						kervalue_evaluate(ker, temp2, ns, es_c, es_beta);
						kervalue = kervalue * ker;

						kervalue_evaluate(ker, temp3, ns, es_c, es_beta);
						kervalue = kervalue * ker;
						// if(outidx==616)printf("%lf,%lu,%d,%d,%d\n",x[k],idx,cur_hive_x,cur_hive_y,cur_hive_z);
						
						// if(outidx==nf1*nf2-1)printf("%lf,%lf,%lf\n",x[k],temp,kervalue);
						sh_fw[threadIdx.x].x += c[k].x * kervalue;
						sh_fw[threadIdx.x].y += c[k].y * kervalue;
						
						// fw[outidx].x += c[k].x * kervalue;
						// fw[outidx].y += c[k].y * kervalue;
						// if(outidx==nf1*nf2*nf3-10)printf("%lf\n",kervalue);

					// 	// if(idx == (nf1*nf2*nf3 - 1))printf("error\n");
						
						
					// 	// sh_fw[threadIdx.x].x += c[k].x * kervalue;
					// 	// sh_fw[threadIdx.x].y += c[k].y * kervalue;
					}
				}
			}
			
			// idx to global mem
			// int outidx = bin_x + bin_y * nf1 + bin_z * nf1 * nf2;
			fw[outidx].x = sh_fw[threadIdx.x].x;
			fw[outidx].y = sh_fw[threadIdx.x].y;
		}
		__syncthreads();
	}
}



// __global__ void partial_3d_conv_sorted()
// {
// 	/*
// 		Based on the image size, do paritical convolution in order to save memory. 
// 		All the parts consists of the whole conv.
// 	*/
// }
