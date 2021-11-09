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

__device__ __constant__ PCS c0[NUM_SEGMENT];
__device__ __constant__ PCS c1[NUM_SEGMENT+3];
__device__ __constant__ PCS c2[NUM_SEGMENT];
__device__ __constant__ PCS c3[NUM_SEGMENT];

void set_ker_eval_lut(PCS *h_c0, PCS *h_c1, PCS *h_c2, PCS *h_c3){
	cudaMemcpyToSymbol(c0, h_c0, NUM_SEGMENT * sizeof(PCS));
	cudaMemcpyToSymbol(c1, h_c1, NUM_SEGMENT * sizeof(PCS));
	cudaMemcpyToSymbol(c2, h_c2, NUM_SEGMENT * sizeof(PCS));
	cudaMemcpyToSymbol(c3, h_c3, NUM_SEGMENT * sizeof(PCS));
}

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

// static __inline__ __device__
// void eval_kernel_vec_Horner(PCS *ker, const PCS x, const int w,
// 	const double upsampfac)
// 	/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
// 	   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
// 	   This is the current evaluation method, since it's faster (except i7 w=16).
// 	   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
// {
// 	PCS z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
// 	// insert the auto-generated code which expects z, w args, writes to ker...
// 	if (upsampfac==2.0) {     // floating point equality is fine here
// #include "../../contrib/ker_horner_allw_loop.c"
// 	}
// }

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
					for(int k=hive_count[cur_hive_idx]; k<hive_count[cur_hive_idx+1]; k++){ 
						// kernel evaluation
						PCS ker;
						PCS kervalue = 1.0;

						PCS temp1 = SHIFT_RESCALE(x[k], nf1, pirange); //save
						temp1 = abs(temp1-bin_x);
						//++++ break if not in range
						if(temp1>nf1/2.0)temp1 = nf1 - temp1;
						if(abs(temp1)>ns/2.0)continue; 

						PCS temp2 = SHIFT_RESCALE(y[k], nf2, pirange);
						temp2 = abs(temp2-bin_y);
						if(temp2>nf2/2.0)temp2 = nf2 - temp2;
						if(abs(temp2)>ns/2.0)continue;

						PCS temp3 = SHIFT_RESCALE(z[k], nf3, pirange);
						temp3 = abs(temp3-bin_z);
						if(temp3>nf3/2.0)temp3 = nf3 - temp3;
						if(abs(temp3)>ns/2.0)continue;
						ker = (abs(temp1) >= ns / 2.0) ? 0.0 : exp(es_beta * (sqrt(1.0 - es_c * temp1  * temp1 )-1));
						// if(outidx==575)printf("1st %.12lf, %lf\n",ker,temp1);

						// kervalue_evaluate(ker, temp, ns, es_c, es_beta);
						kervalue = kervalue * ker;

						ker = (abs(temp2) >= ns / 2.0) ? 0.0 : exp(es_beta * (sqrt(1.0 - es_c * temp2  * temp2 )-1));
						// if(outidx==575)printf("2nd %.12lf\n",ker);

						// kervalue_evaluate(ker, temp2, ns, es_c, es_beta);
						kervalue = kervalue * ker;
						ker = (abs(temp3) >= ns / 2.0) ? 0.0 : exp(es_beta * (sqrt(1.0 - es_c * temp3  * temp3 )-1));

						// kervalue_evaluate(ker, temp3, ns, es_c, es_beta);
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


__global__ void conv_3d_outputdriven_shared_sparse(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
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
		__shared__ PCS sh_x[SHARED_SIZE_3D_HIVE];
		__shared__ PCS sh_y[SHARED_SIZE_3D_HIVE];
		__shared__ PCS sh_z[SHARED_SIZE_3D_HIVE];
		__shared__ CUCPX sh_c[SHARED_SIZE_3D_HIVE];
		__shared__ int neighbor_info[27];

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
		
		int flag = 0; // first bit is for x, y, z later consider this issue
	
		// start_hive_idx[1] = cur_hive_idx - nhive_x - 1;
		// start_hive_idx[2] = start_hive_idx[1] + nhive_x*nhive_y;
		// start_hive_idx[0] = start_hive_idx[1] - nhive_x*nhive_y; 
		
		if(threadIdx.x<27){ // have a litter improvement
			int cur_hive_x;
			int cur_hive_y;
			int cur_hive_z; 
			
			cur_hive_z = hive_z + threadIdx.x / 9 - 1;
			cur_hive_y = hive_y + threadIdx.x % 9 / 3 - 1;
			cur_hive_x = hive_x + threadIdx.x % 3 - 1;

			// some issues here
			if(cur_hive_x >= nhive_x || cur_hive_x < 0)  cur_hive_x -= ((cur_hive_x > 0) - (cur_hive_x < 0))*nhive_x;
			if(cur_hive_y >= nhive_y || cur_hive_y < 0)  cur_hive_y -= ((cur_hive_y > 0) - (cur_hive_y < 0))*nhive_y;
			if(cur_hive_z >= nhive_z || cur_hive_z < 0)  cur_hive_z -= ((cur_hive_z > 0) - (cur_hive_z < 0))*nhive_z;

			neighbor_info[threadIdx.x] = cur_hive_x + cur_hive_y * nhive_x + cur_hive_z * nhive_x * nhive_y;
		}
		__syncthreads();
		
		// loop from here
		int hive_index = 0;
		while(hive_index<27){
			if(flag>=0)flag = 0;
			cur_hive_idx = 0; // reuse as start of shared memory
			// load data into shared memroy
			for(; hive_index<27; hive_index++){
				// if flag = -1, cur_nupt_num changed
				int cur_nupt_num;
				if(flag<0)
				cur_nupt_num = hive_count[neighbor_info[hive_index]+1]+flag*SHARED_SIZE_3D_HIVE;
				else
				cur_nupt_num = hive_count[neighbor_info[hive_index]+1]-hive_count[neighbor_info[hive_index]];
				// if(threadIdx.x==0&&blockIdx.x==0)printf("number of point in hive %d: %d\n",hive_index,cur_nupt_num);
				if(cur_hive_idx+cur_nupt_num<=SHARED_SIZE_3D_HIVE){
					// load to shared mem
					flag = hive_count[neighbor_info[hive_index]]; //reuse flag
					for(int j = threadIdx.x; j<cur_nupt_num; j+=blockDim.x){
						// +++ shift here
						// sh_x[cur_hive_idx+j] = SHIFT_RESCALE(x[flag+j], nf1, pirange);
						// sh_y[cur_hive_idx+j] = SHIFT_RESCALE(y[flag+j], nf2, pirange);
						// sh_z[cur_hive_idx+j] = SHIFT_RESCALE(z[flag+j], nf3, pirange);
						sh_x[cur_hive_idx+j] = x[flag+j];
						sh_y[cur_hive_idx+j] = y[flag+j];
						sh_z[cur_hive_idx+j] = z[flag+j];
						sh_c[cur_hive_idx+j] = c[flag+j]; // save those shifted stuff
					}
					cur_hive_idx+=cur_nupt_num;
				}
				else{
					// points in one hive can not load into shared mem
					if(cur_hive_idx==0){
						// fully occupy the shared mem
						// printf("1 \n");
						int start_idx_full = hive_count[neighbor_info[hive_index]] - flag * SHARED_SIZE_3D_HIVE;
						for(int j = threadIdx.x; j<SHARED_SIZE_3D_HIVE; j+=blockDim.x){
							// +++ shift here
							// sh_x[j] = SHIFT_RESCALE(x[start_idx_full+j], nf1, pirange);
							// sh_y[j] = SHIFT_RESCALE(y[start_idx_full+j], nf2, pirange);
							// sh_z[j] = SHIFT_RESCALE(z[start_idx_full+j], nf3, pirange);
							sh_x[j] = x[start_idx_full+j];
							sh_y[j] = y[start_idx_full+j];
							sh_z[j] = z[start_idx_full+j];
							sh_c[j] = c[start_idx_full+j];
						}
						cur_hive_idx = SHARED_SIZE_3D_HIVE;
						// hive_index--;
						flag--;
					}
					// hive_index++;
					break;
				}
			}
			__syncthreads();

			if(bin_x<nf1&&bin_y<nf2&&bin_z<nf3){
				for(int i=0; i<cur_hive_idx; i++){
					
					// kernel evaluation
					PCS ker;
					PCS kervalue = 1.0;

					PCS temp1 = abs(sh_x[i]-bin_x);
					//++++ break if not in range
					if(temp1>nf1/2.0)temp1 = abs(nf1 - temp1);
					if(temp1>=ns/2.0)continue; 

					PCS temp2 = abs(sh_y[i]-bin_y);
					if(temp2>nf2/2.0)temp2 = abs(nf2 - temp2);
					if(temp2>=ns/2.0)continue;

					PCS temp3 = abs(sh_z[i]-bin_z);
					if(temp3>nf3/2.0)temp3 = abs(nf3 - temp3);
					if(temp3>=ns/2.0)continue;

					// if(outidx==0)printf("%lf,%lf,%lf,%lf\n",temp,temp2,temp3,c[k].x);
					
					ker = exp(es_beta * (sqrt(1.0 - es_c * temp1  * temp1 )));
					// kervalue_evaluate(ker, temp, ns, es_c, es_beta);
					kervalue = kervalue * ker;
					ker = exp(es_beta * (sqrt(1.0 - es_c * temp2  * temp2 )));
					// kervalue_evaluate(ker, temp2, ns, es_c, es_beta);
					kervalue = kervalue * ker;
					ker = exp(es_beta * (sqrt(1.0 - es_c * temp3  * temp3 )));
					// kervalue_evaluate(ker, temp3, ns, es_c, es_beta);
					kervalue = kervalue * ker;
					
					// if(outidx==616)printf("%lf,%lu,%d,%d,%d\n",x[k],idx,cur_hive_x,cur_hive_y,cur_hive_z);
					
					// if(outidx==nf1*nf2-1)printf("%lf,%lf,%lf\n",x[k],temp,kervalue);
					fw[outidx].x += sh_c[i].x * kervalue;
					fw[outidx].y += sh_c[i].y * kervalue;
				
				}
			}
			__syncthreads();
		}
	}
}

__global__ void conv_3d_outputdriven_shared_hive_lut(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, PCS *c0, int* hive_count, const int ns, int nf1, int nf2,
	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, int pirange){
	/*
		blocksize = 8*8*8 if change may need to revise
		another method also load intput into shared memroy by multi times

		remove some variable or put to constant memory remove nbin
	*/
	
	unsigned long int idx; // one hive by one hive
	unsigned long int M = nbin_x; // the threads are padded
	M *= nbin_y;
	M *= nbin_z;
	double ns_2 = 2 / (double) ns;
	double seg_s = ns_2 * SHARED_SIZE_SEG;
	double num_s_1 = 1 / (double) SHARED_SIZE_SEG;
	__shared__ PCS sh_c0[SHARED_SIZE_SEG*SEG_ORDER];
	for(idx = threadIdx.x; idx<SHARED_SIZE_SEG*SEG_ORDER; idx+=blockDim.x){
		sh_c0[idx] = c0[idx];
	}
	__syncthreads();
	for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < M; idx += blockDim.x * gridDim.x)
	{
		int hive_x, hive_y, hive_z;
		unsigned long int outidx;
		// int bin_idx;
		// load to shared memory __synchronize
		// extern __shared__ CUCPX sh_fw[];
		__shared__ PCS sh_x[SHARED_SIZE_SEG];
		__shared__ PCS sh_y[SHARED_SIZE_SEG];
		__shared__ PCS sh_z[SHARED_SIZE_SEG];
		__shared__ CUCPX sh_c[SHARED_SIZE_SEG];
		__shared__ int neighbor_info[27];

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
		
		int flag = 0; // first bit is for x, y, z later consider this issue
	
		
		if(threadIdx.x<27){ // have a litter improvement
			int cur_hive_x;
			int cur_hive_y;
			int cur_hive_z; 
			
			cur_hive_z = hive_z + threadIdx.x / 9 - 1;
			cur_hive_y = hive_y + threadIdx.x % 9 / 3 - 1;
			cur_hive_x = hive_x + threadIdx.x % 3 - 1;

			// some issues here if nhive<3 we will not adopt this method
			if(cur_hive_x >= nhive_x || cur_hive_x < 0)  cur_hive_x -= ((cur_hive_x > 0) - (cur_hive_x < 0))*nhive_x;
			if(cur_hive_y >= nhive_y || cur_hive_y < 0)  cur_hive_y -= ((cur_hive_y > 0) - (cur_hive_y < 0))*nhive_y;
			if(cur_hive_z >= nhive_z || cur_hive_z < 0)  cur_hive_z -= ((cur_hive_z > 0) - (cur_hive_z < 0))*nhive_z;

			neighbor_info[threadIdx.x] = cur_hive_x + cur_hive_y * nhive_x + cur_hive_z * nhive_x * nhive_y;
		}
		__syncthreads();
		
		// loop from here
		int hive_index = 0;
		while(hive_index<27){
			if(flag>=0)flag = 0;
			cur_hive_idx = 0; // reuse as start of shared memory
			// load data into shared memroy
			for(; hive_index<27; hive_index++){
				// if flag = -1, cur_nupt_num changed
				int cur_nupt_num;
				if(flag<0)
				cur_nupt_num = hive_count[neighbor_info[hive_index]+1]+flag*SHARED_SIZE_SEG;
				else
				cur_nupt_num = hive_count[neighbor_info[hive_index]+1]-hive_count[neighbor_info[hive_index]];
				// if(threadIdx.x==0&&blockIdx.x==0)printf("number of point in hive %d: %d\n",hive_index,cur_nupt_num);
				if(cur_hive_idx+cur_nupt_num<=SHARED_SIZE_SEG){
					// load to shared mem
					flag = hive_count[neighbor_info[hive_index]]; //reuse flag
					for(int j = threadIdx.x; j<cur_nupt_num; j+=blockDim.x){
						// +++ shift here
						sh_x[cur_hive_idx+j] = x[flag+j];
						sh_y[cur_hive_idx+j] = y[flag+j];
						sh_z[cur_hive_idx+j] = z[flag+j];
						sh_c[cur_hive_idx+j] = c[flag+j]; // save those shifted stuff
					}
					cur_hive_idx+=cur_nupt_num;
				}
				else{
					// points in one hive can not load into shared mem
					if(cur_hive_idx==0){
						// fully occupy the shared mem
						// printf("1 \n");
						int start_idx_full = hive_count[neighbor_info[hive_index]] - flag * SHARED_SIZE_SEG;
						for(int j = threadIdx.x; j<SHARED_SIZE_SEG; j+=blockDim.x){
							// +++ shift here
							sh_x[j] = x[start_idx_full+j];
							sh_y[j] = y[start_idx_full+j];
							sh_z[j] = z[start_idx_full+j];
							sh_c[j] = c[start_idx_full+j];
						}
						cur_hive_idx = SHARED_SIZE_SEG;
						// hive_index--;
						flag--;
					}
					// hive_index++;
					break;
				}
			}
			__syncthreads();

			if(bin_x<nf1&&bin_y<nf2&&bin_z<nf3){
				for(int i=0; i<cur_hive_idx; i++){
					
					// kernel evaluation
					// PCS ker;
					PCS kervalue = 1.0;

					PCS temp1 = abs(sh_x[i]-bin_x);
					//++++ break if not in range
					if(temp1>nf1/2.0)temp1 = abs(nf1 - temp1);
					// if(outidx==575&&i==491)printf("temp: %.6g\n",temp1);
					if(temp1>=ns/2.0)continue; 

					// PCS temp2 = abs(sh_y[i]-bin_y);
					// if(temp2>nf2/2.0)temp2 = abs(nf2 - temp2);
					// if(temp2>=ns/2.0)continue;

					// PCS temp3 = abs(sh_z[i]-bin_z);
					// if(temp3>nf3/2.0)temp3 = abs(nf3 - temp3);
					// if(temp3>=ns/2.0)continue;

					// if(outidx==3)printf("temp: %lf\n",temp1);
					
					int seg_idx = temp1 * seg_s;
					double dis = temp1 * ns_2 - num_s_1 * seg_idx;
					seg_idx *= SEG_ORDER;
					kervalue =sh_c0[seg_idx] + dis*(sh_c0[seg_idx+1] + dis*(sh_c0[seg_idx+2] + dis*(sh_c0[seg_idx+3]+dis*sh_c0[seg_idx+4])));
					
					temp1 = abs(sh_y[i]-bin_y); // it will be faster just use one variable?
					if(temp1>nf2/2.0)temp1 = abs(nf2 - temp1);
					if(temp1>=ns/2.0)continue;
					seg_idx = temp1 * seg_s;
					dis = temp1 * ns_2 - num_s_1 * seg_idx;
					seg_idx *= SEG_ORDER;
					kervalue *=sh_c0[seg_idx] + dis*(sh_c0[seg_idx+1] + dis*(sh_c0[seg_idx+2] + dis*(sh_c0[seg_idx+3]+dis*sh_c0[seg_idx+4])));
					
					temp1 = abs(sh_z[i]-bin_z);
					if(temp1>nf3/2.0)temp1 = abs(nf3 - temp1);
					if(temp1>=ns/2.0)continue;
					seg_idx = temp1 * seg_s;
					dis = temp1 * ns_2 - num_s_1 * seg_idx;
					seg_idx *= SEG_ORDER;
					kervalue *=sh_c0[seg_idx] + dis*(sh_c0[seg_idx+1] + dis*(sh_c0[seg_idx+2] + dis*(sh_c0[seg_idx+3]+dis*sh_c0[seg_idx+4])));
					// if(outidx==575)printf("temp: %.6g\n",kervalue);
					// if(outidx==575)printf("temp: %.12lf\n",kervalue);
					// if(outidx==3)printf("%d, %lf, %lf\n",seg_idx, sh_c0[seg_idx], kervalue);
					// seg_idx = temp2 * seg_s;
					// dis = temp2 * ns_2 - num_s_1 * seg_idx;
					// seg_idx *= SEG_ORDER;
					// kervalue *=sh_c0[seg_idx] + dis*(sh_c0[seg_idx+1] + dis*(sh_c0[seg_idx+2] + dis*(sh_c0[seg_idx+3]+dis*sh_c0[seg_idx+4])));
					
					// // kervalue *= c1[seg_idx]; 
					// // if(outidx==3)printf("%d, %lf\n",seg_idx, c0[seg_idx] + dis*(c1[seg_idx] + dis*(c2[seg_idx] + dis*c3[seg_idx])));

					// seg_idx = temp3 * seg_s;
					// dis = temp3 * ns_2 - num_s_1 * seg_idx;
					// seg_idx *= SEG_ORDER;
					// kervalue *=sh_c0[seg_idx] + dis*(sh_c0[seg_idx+1] + dis*(sh_c0[seg_idx+2] + dis*(sh_c0[seg_idx+3]+dis*sh_c0[seg_idx+4])));
					// if(outidx==616)printf("%lf,%lu,%d,%d,%d\n",x[k],idx,cur_hive_x,cur_hive_y,cur_hive_z);
					
					// if(outidx==nf1*nf2-1)printf("%lf,%lf,%lf\n",x[k],temp,kervalue);
					fw[outidx].x += sh_c[i].x * kervalue;
					fw[outidx].y += sh_c[i].y * kervalue;
				
				}
			}
			__syncthreads();
		}
	}
}


__global__ void conv_3d_outputdriven_shared_hive_lut_constant(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, const int ns, int nf1, int nf2,
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
	double ns_2 = 2 / (double) ns;
	double seg_s = ns_2 * NUM_SEGMENT;
	double num_s_1 = 1 / (double) NUM_SEGMENT;
	for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < M; idx += blockDim.x * gridDim.x)
	{
		int hive_x, hive_y, hive_z;
		unsigned long int outidx;
		// int bin_idx;
		// load to shared memory __synchronize
		// extern __shared__ CUCPX sh_fw[];
		__shared__ PCS sh_x[SHARED_SIZE_3D_HIVE];
		__shared__ PCS sh_y[SHARED_SIZE_3D_HIVE];
		__shared__ PCS sh_z[SHARED_SIZE_3D_HIVE];
		__shared__ CUCPX sh_c[SHARED_SIZE_3D_HIVE];
		__shared__ int neighbor_info[27];

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
		
		int flag = 0; // first bit is for x, y, z later consider this issue
	
		// start_hive_idx[1] = cur_hive_idx - nhive_x - 1;
		// start_hive_idx[2] = start_hive_idx[1] + nhive_x*nhive_y;
		// start_hive_idx[0] = start_hive_idx[1] - nhive_x*nhive_y; 
		
		if(threadIdx.x<27){ // have a litter improvement
			int cur_hive_x;
			int cur_hive_y;
			int cur_hive_z; 
			
			cur_hive_z = hive_z + threadIdx.x / 9 - 1;
			cur_hive_y = hive_y + threadIdx.x % 9 / 3 - 1;
			cur_hive_x = hive_x + threadIdx.x % 3 - 1;

			// some issues here
			if(cur_hive_x >= nhive_x || cur_hive_x < 0) nhive_x<3? flag=1: cur_hive_x -= ((cur_hive_x > 0) - (cur_hive_x < 0))*nhive_x;
			if(cur_hive_y >= nhive_y || cur_hive_y < 0) nhive_y<3? flag=1: cur_hive_y -= ((cur_hive_y > 0) - (cur_hive_y < 0))*nhive_y;
			if(cur_hive_z >= nhive_z || cur_hive_z < 0) nhive_z<3? flag=1: cur_hive_z -= ((cur_hive_z > 0) - (cur_hive_z < 0))*nhive_z;

			neighbor_info[threadIdx.x] = cur_hive_x + cur_hive_y * nhive_x + cur_hive_z * nhive_x * nhive_y;
		}
		__syncthreads();
		
		// loop from here
		int hive_index = 0;
		while(hive_index<27){
			if(flag>=0)flag = 0;
			cur_hive_idx = 0; // reuse as start of shared memory
			// load data into shared memroy
			for(; hive_index<27; hive_index++){
				// if flag = -1, cur_nupt_num changed
				int cur_nupt_num;
				if(flag<0)
				cur_nupt_num = hive_count[neighbor_info[hive_index]+1]+flag*SHARED_SIZE_3D_HIVE;
				else
				cur_nupt_num = hive_count[neighbor_info[hive_index]+1]-hive_count[neighbor_info[hive_index]];
				// if(threadIdx.x==0&&blockIdx.x==0)printf("number of point in hive %d: %d\n",hive_index,cur_nupt_num);
				if(cur_hive_idx+cur_nupt_num<=SHARED_SIZE_3D_HIVE){
					// load to shared mem
					flag = hive_count[neighbor_info[hive_index]]; //reuse flag
					for(int j = threadIdx.x; j<cur_nupt_num; j+=blockDim.x){
						// +++ shift here
						sh_x[cur_hive_idx+j] = x[flag+j];
						sh_y[cur_hive_idx+j] = y[flag+j];
						sh_z[cur_hive_idx+j] = z[flag+j];
						sh_c[cur_hive_idx+j] = c[flag+j]; // save those shifted stuff
					}
					cur_hive_idx+=cur_nupt_num;
				}
				else{
					// points in one hive can not load into shared mem
					if(cur_hive_idx==0){
						// fully occupy the shared mem
						// printf("1 \n");
						int start_idx_full = hive_count[neighbor_info[hive_index]] - flag * SHARED_SIZE_3D_HIVE;
						for(int j = threadIdx.x; j<SHARED_SIZE_3D_HIVE; j+=blockDim.x){
							// +++ shift here
							// sh_x[j] = SHIFT_RESCALE(x[start_idx_full+j], nf1, pirange);
							// sh_y[j] = SHIFT_RESCALE(y[start_idx_full+j], nf2, pirange);
							// sh_z[j] = SHIFT_RESCALE(z[start_idx_full+j], nf3, pirange);
							sh_x[j] = x[start_idx_full+j];
							sh_y[j] = y[start_idx_full+j];
							sh_z[j] = z[start_idx_full+j];
							sh_c[j] = c[start_idx_full+j];
						}
						cur_hive_idx = SHARED_SIZE_3D_HIVE;
						hive_index--;
						flag--;
					}
					hive_index++;
					break;
				}
			}
			__syncthreads();

			if(bin_x<nf1&&bin_y<nf2&&bin_z<nf3){
				for(int i=0; i<cur_hive_idx; i++){
					
					// kernel evaluation
					PCS ker;
					PCS kervalue = 1.0;

					PCS temp1 = abs(sh_x[i]-bin_x);
					//++++ break if not in range
					if(temp1>nf1/2.0)temp1 = abs(nf1 - temp1);
					if(temp1>=ns/2.0)continue; 

					PCS temp2 = abs(sh_y[i]-bin_y);
					if(temp2>nf2/2.0)temp2 = abs(nf2 - temp2);
					if(temp2>=ns/2.0)continue;

					PCS temp3 = abs(sh_z[i]-bin_z);
					if(temp3>nf3/2.0)temp3 = abs(nf3 - temp3);
					if(temp3>=ns/2.0)continue;

					// if(outidx==3)printf("temp: %lf,%lf,%lf, %d\n",temp1,temp2,temp3,ns);
					
					int seg_idx = temp1 * seg_s;
					double dis = temp1 * ns_2 - num_s_1 * seg_idx;
					kervalue =c0[seg_idx] + dis*(c1[seg_idx] + dis*(c2[seg_idx] + dis*c3[seg_idx]));
					// if(outidx==3)printf("%d, %lf, %lf\n",seg_idx, c0[seg_idx], kervalue); + dis*(c1[seg_idx] + dis*(c2[seg_idx] + dis*c3[seg_idx]))
					seg_idx = temp2 * seg_s;
					dis = temp2 * ns_2 - num_s_1 * seg_idx;
					kervalue *=c0[seg_idx] + dis*(c1[seg_idx] + dis*(c2[seg_idx] + dis*c3[seg_idx]));
					// kervalue *= c1[seg_idx]; 
					// if(outidx==3)printf("%d, %lf\n",seg_idx, c0[seg_idx] + dis*(c1[seg_idx] + dis*(c2[seg_idx] + dis*c3[seg_idx])));

					seg_idx = temp3 * seg_s;
					dis = temp3 * ns_2 - num_s_1 * seg_idx;
					kervalue *=c0[seg_idx] + dis*(c1[seg_idx] + dis*(c2[seg_idx] + dis*c3[seg_idx]));
					// kervalue *= c1[seg_idx];
					// if(outidx==3)printf("%d, %lf\n",seg_idx, c0[seg_idx] + dis*(c1[seg_idx] + dis*(c2[seg_idx] + dis*c3[seg_idx])));

					// printf("%d, %lf\n",seg_idx, c0[seg_idx]);
					// ker = exp(es_beta * (sqrt(1.0 - es_c * temp1  * temp1 )));
					// kervalue *= ker;
					// ker = exp(es_beta * (sqrt(1.0 - es_c * temp2  * temp2 )));
					// kervalue *= ker;
					// ker = exp(es_beta * (sqrt(1.0 - es_c * temp3  * temp3 )));
					// kervalue *= ker;
					
					// if(outidx==616)printf("%lf,%lu,%d,%d,%d\n",x[k],idx,cur_hive_x,cur_hive_y,cur_hive_z);
					
					// if(outidx==nf1*nf2-1)printf("%lf,%lf,%lf\n",x[k],temp,kervalue);
					fw[outidx].x += sh_c[i].x * kervalue;
					fw[outidx].y += sh_c[i].y * kervalue;
				
				}
			}
			__syncthreads();
		}
	}
}

__global__ void conv_3d_outputdriven_shared(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int* hive_count, unsigned short int *n_share, const int ns, int nf1, int nf2,
	 int nf3, int nbin_x, int nbin_y, int nbin_z, int nhive_x, int nhive_y, int nhive_z, PCS es_c, PCS es_beta, int pirange){
	/*
		blocksize = 8*8*8 if change may need to revise,,,,,, DO NOT USE HIVE
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
		__shared__ PCS sh_x[SHARED_SIZE_3D_HIVE];
		__shared__ PCS sh_y[SHARED_SIZE_3D_HIVE];
		__shared__ PCS sh_z[SHARED_SIZE_3D_HIVE];
		__shared__ CUCPX sh_fw[SHARED_SIZE_3D_HIVE];

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
		
		int hive_index = 0;
		for(int start_s=0; start_s<n_share[blockIdx.x]; start_s++){
			// load uvw and fw into shared memory
			// one hive by one hive


		} // not the last

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
					for(int k=hive_count[cur_hive_idx]; k<hive_count[cur_hive_idx+1]; k++){ 
						// kernel evaluation
						PCS ker;
						PCS kervalue = 1.0;

						PCS temp1 = SHIFT_RESCALE(x[k], nf1, pirange); //save
						temp1 = abs(temp1-bin_x);
						//++++ break if not in range
						if(temp1>nf1/2.0)temp1 = nf1 - temp1;
						if(abs(temp1)>ns/2.0)continue; 

						PCS temp2 = SHIFT_RESCALE(y[k], nf2, pirange);
						temp2 = abs(temp2-bin_y);
						if(temp2>nf2/2.0)temp2 = nf2 - temp2;
						if(abs(temp2)>ns/2.0)continue;

						PCS temp3 = SHIFT_RESCALE(z[k], nf3, pirange);
						temp3 = abs(temp3-bin_z);
						if(temp3>nf3/2.0)temp3 = nf3 - temp3;
						if(abs(temp3)>ns/2.0)continue;
						// if(outidx==0)printf("%lf,%lf,%lf,%lf\n",temp,temp2,temp3,c[k].x);
						ker = (abs(temp1) >= ns / 2.0) ? 0.0 : exp(es_beta * (sqrt(1.0 - es_c * temp1  * temp1 )));
						// kervalue_evaluate(ker, temp, ns, es_c, es_beta);
						kervalue = kervalue * ker;
						ker = (abs(temp2) >= ns / 2.0) ? 0.0 : exp(es_beta * (sqrt(1.0 - es_c * temp2  * temp2 )));
						// kervalue_evaluate(ker, temp2, ns, es_c, es_beta);
						kervalue = kervalue * ker;

						ker = (abs(temp3) >= ns / 2.0) ? 0.0 : exp(es_beta * (sqrt(1.0 - es_c * temp3  * temp3 )));
						// kervalue_evaluate(ker, temp3, ns, es_c, es_beta);
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


// __global__ void partial_3d_conv_sorted()
// {
// 	/*
// 		Based on the image size, do paritical convolution in order to save memory. 
// 		All the parts consists of the whole conv.
// 	*/
// }
