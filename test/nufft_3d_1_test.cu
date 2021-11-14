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
	/* Input: M, N1, N2, N3 epsilon method
		method - conv method
		M - number of randomly distributed points
		N1, N2 - output size
		epsilon - tolerance
	*/
	int ier = 0;
	if (argc < 5)
	{
		fprintf(stderr,
				"Usage: W Stacking\n"
				"Arguments:\n"
				"  N1, N2 : image size.\n"
				"  M: The number of randomly distributed points.\n"
				"  epsilon: NUFFT tolerance (default 1e-6).\n"
				"  kerevalmeth: Kernel evaluation method; one of\n"
				"     0: Exponential of square root (default), or\n"
				"     1: Horner evaluation.\n"
				"  method: One of\n"
				"    0: nupts driven (default),\n"
				"    2: upts driven sort, or\n");
		return 1;
	}
	int N1, N2, N3;
	PCS sigma = 2.0; // upsampling factor
	int M;

	double inp;
	sscanf(argv[1], "%d", &N1);
	sscanf(argv[2], "%d", &N2);
	sscanf(argv[3], "%d", &N3);
	sscanf(argv[4], "%d", &M);
	PCS epsilon = 1e-10;
	if(argc>4){
		sscanf(argv[5], "%lf", &inp);
		epsilon = inp;
	}
	int kerevalmeth = 0;
	if(argc>5)sscanf(argv[7], "%d", &kerevalmeth);
	int method=2;
	if(argc>6)sscanf(argv[8], "%d", &method);

	//gpu_method == 0, nupts driven

	//int ier;
	PCS *u, *v, *w;
	CPX *c;
	u = (PCS *)malloc(M * sizeof(PCS)); //Allocates page-locked memory on the host.
	v = (PCS *)malloc(M * sizeof(PCS));
	w = (PCS *)malloc(M * sizeof(PCS));
	c = (CPX *)malloc(M * sizeof(CPX));
	PCS *d_u, *d_v, *d_w;
	CUCPX *d_c, *d_fk;
	// CUCPX *d_fw;
	checkCudaErrors(cudaMalloc(&d_u, M * sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_v, M * sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_w, M * sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_c, M * sizeof(CUCPX)));

	// generating data
	for (int i = 0; i < M; i++)
	{
		u[i] = randm11()*PI; //xxxxx
		v[i] = randm11()*PI;
		w[i] = randm11()*PI;
		c[i].real(randm11()); // M vis per channel, weight?
		c[i].imag(randm11());
		// wgt[i] = 1;
	}

	// double a[5] = {-PI/2, -PI/3, 0, PI/3, PI/2}; // change to random data
	// for(int i=0; i<M; i++){
	// 	u[i] = a[i/5];
	// 	v[i] = a[i%5];
	// }
#ifdef DEBUG
	printf("origial input data...\n");
	for(int i=0; i<M; i++){
		printf("%.3lf ",u[i]);
	}
	printf("\n");
	for(int i=0; i<M; i++){
		printf("%.3lf ",c[i].real());
	}
	printf("\n");
#endif
	// ignore the tdirty
	// how to convert ms to vis

	//printf("generated data, x[1] %2.2g, y[1] %2.2g , z[1] %2.2g, c[1] %2.2g\n",x[1] , y[1], z[1], c[1].real());

	// Timing begin
	cudaEvent_t cuda_start, cuda_end;

	float kernel_time;

	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_end);

	cudaEventRecord(cuda_start);

	//data transfer
	checkCudaErrors(cudaMemcpy(d_u, u, M * sizeof(PCS), cudaMemcpyHostToDevice)); //u
	checkCudaErrors(cudaMemcpy(d_v, v, M * sizeof(PCS), cudaMemcpyHostToDevice)); //v
	checkCudaErrors(cudaMemcpy(d_w, w, M * sizeof(PCS), cudaMemcpyHostToDevice)); //v
	checkCudaErrors(cudaMemcpy(d_c, c, M * sizeof(CUCPX), cudaMemcpyHostToDevice));

	/* ----------Step2: plan setting------------*/
	CURAFFT_PLAN *plan;

	plan = new CURAFFT_PLAN();
    memset(plan, 0, sizeof(CURAFFT_PLAN));

	int direction = 1; //inverse
	
	// opts and copts setting
    plan->opts.gpu_device_id = 0;
    plan->opts.upsampfac = sigma;
    plan->opts.gpu_sort = 1;
    plan->opts.gpu_binsizex = -1;
    plan->opts.gpu_binsizey = -1;
    plan->opts.gpu_binsizez = -1;
    plan->opts.gpu_kerevalmeth = kerevalmeth;
    plan->opts.gpu_conv_only = 0;
    plan->opts.gpu_gridder_method = method;

    ier = setup_conv_opts(plan->copts, epsilon, sigma, 1, direction, kerevalmeth); //check the arguements

	if(ier!=0)printf("setup_error\n");

	if(kerevalmeth==1){
        PCS *h_c0 = (PCS *)malloc(sizeof(PCS)*SEG_ORDER*SHARED_SIZE_SEG);
        taylor_series_approx_factors(h_c0,plan->copts.ES_beta,SHARED_SIZE_SEG,SEG_ORDER,kerevalmeth);
		checkCudaErrors(cudaMalloc((void**)&plan->c0,sizeof(PCS)*SEG_ORDER*SHARED_SIZE_SEG));
		checkCudaErrors(cudaMemcpy(plan->c0,h_c0,sizeof(PCS)*SEG_ORDER*SHARED_SIZE_SEG,cudaMemcpyHostToDevice));
        free(h_c0);
    }

    // plan setting
    // cuda stream malloc in setup_plan
    

    int nf1 = get_num_cells(N1,plan->copts);
    int nf2 = get_num_cells(N2,plan->copts);
    int nf3 = get_num_cells(N3,plan->copts);
    // printf("nf3 %d\n",nf3);
    plan->dim = 3;
	plan->type = 1;
	// show_mem_usage();
    setup_plan(nf1, nf2, nf3, M, d_u, d_v, d_w, d_c, plan);
	// show_mem_usage();

	plan->ms = N1;
	plan->mt = N2;
	plan->mu = N3;
    plan->execute_flow = 1;
	int iflag = direction;
    int fftsign = (iflag>=0) ? 1 : -1;

	plan->iflag = fftsign; //may be useless| conflict with direction
	plan->batchsize = 1;

    plan->copts.direction = direction; // 1 inverse, 0 forward


    // // copy to device 
    // checkCudaErrors(cudaMemcpy(plan->fwkerhalf1,fwkerhalf1,(plan->nf1/2+1)*
	// 	sizeof(PCS),cudaMemcpyHostToDevice));
	
	// checkCudaErrors(cudaMemcpy(plan->fwkerhalf2,fwkerhalf2,(plan->nf2/2+1)*
	// 	sizeof(PCS),cudaMemcpyHostToDevice));
    
    // cufft plan setting
    

    // set up bin size +++ (for other methods) and related malloc based on gpu method
    // assign memory for index after sorting (can be done in setup_plan)
    // bin sorting (for other methods)

	if(ier == 1){
		printf("errors in gridder setting\n");
		return ier;
	}
	// fw (conv res set)
	// checkCudaErrors(cudaMalloc((void**)&d_fw,sizeof(CUCPX)*nf1*nf2));
	// checkCudaErrors(cudaMemset(d_fw, 0, sizeof(CUCPX)*nf1*nf2));
	// plan->fw = d_fw;
	// fk malloc and set
	checkCudaErrors(cudaMalloc((void**)&d_fk,sizeof(CUCPX)*N1*N2*N3));
	plan->fk = d_fk;

	// show_mem_usage();
    // prestage
    cura_prestage(plan);

#ifdef DEBUG
	printf("nf1, nf2, nf3 %d %d %d\n",plan->nf1,plan->nf2,plan->nf3);
	printf("copts info printing...\n");
	printf("kw: %d, direction: %d, pirange: %d, upsampfac: %lf, \nbeta: %lf, halfwidth: %lf, c: %lf\n",
 	plan->copts.kw,
 	plan->copts.direction,
 	plan->copts.pirange,
 	plan->copts.upsampfac,
    plan->copts.ES_beta,
    plan->copts.ES_halfwidth,
    plan->copts.ES_c);

	PCS *fwkerhalf1 = (PCS*)malloc(sizeof(PCS)*(plan->nf1/2+1));
	PCS *fwkerhalf2 = (PCS*)malloc(sizeof(PCS)*(plan->nf2/2+1));
    PCS *fwkerhalf3 = (PCS*)malloc(sizeof(PCS)*(plan->nf3/2+1));

	checkCudaErrors(cudaMemcpy(fwkerhalf1,plan->fwkerhalf1,(plan->nf1/2+1)*
	 	sizeof(PCS),cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaMemcpy(fwkerhalf2,plan->fwkerhalf2,(plan->nf2/2+1)*
	 	sizeof(PCS),cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(fwkerhalf3,plan->fwkerhalf3,(plan->nf3/2+1)*
	 	sizeof(PCS),cudaMemcpyDeviceToHost));
	
	printf("correction factor print...\n");
	for(int i=0; i<nf1/2+1; i++){
		printf("%.3g ", fwkerhalf1[i]);
	}
	printf("\n");

	for(int i=0; i<nf2/2+1; i++){
		printf("%.3g ", fwkerhalf2[i]);
	}
	printf("\n");

    for(int i=0; i<nf3/2+1; i++){
		printf("%.3g ", fwkerhalf3[i]);
	}
	printf("\n");
	// free host fwkerhalf
    free(fwkerhalf1);
    free(fwkerhalf2);
    free(fwkerhalf3);
#endif

	// calulating result
	curafft_conv(plan);
#ifdef DEBUG
	printf("conv result printing...\n");
	CPX *fw = (CPX *)malloc(sizeof(CPX)*nf1*nf2*nf3);
	PCS temp_res=0;
	cudaMemcpy(fw,plan->fw,sizeof(CUCPX)*nf1*nf2*nf3,cudaMemcpyDeviceToHost);
	for(int i=0; i<nf2; i++){
		for(int j=0; j<nf1; j++){
			printf("%.3g ",fw[i*nf1+j].real());
			temp_res += fw[i*nf1+j].real();
		}
		printf("\n");
	}
	printf("fft(0,0) %.3g\n",temp_res);
#endif
	cufftHandle fftplan;
    int n[] = {plan->nf3, plan->nf2, plan->nf1};
    int inembed[] = {plan->nf3, plan->nf2, plan->nf1};
	int onembed[] = {plan->nf3, plan->nf2, plan->nf1}; //too many points?
    
	// cufftCreate(&fftplan);
	// cufftPlan2d(&fftplan,n[0],n[1],CUFFT_TYPE);
    // the bach size sets as the num of w when memory is sufficent. Alternative way, set as a smaller number when memory is insufficient.
    // and handle this piece by piece 
	cufftPlanMany(&fftplan,3,n,inembed,1,inembed[0]*inembed[1]*inembed[2],
		onembed,1,onembed[0]*onembed[1],CUFFT_TYPE,1); //need to check and revise (the partial conv will be differnt)
    plan->fftplan = fftplan; 
	// fft
	CUFFT_EXEC(plan->fftplan, plan->fw, plan->fw, direction);
#ifdef DEBUG 
	printf("fft result printing...\n");
	cudaMemcpy(fw,plan->fw,sizeof(CUCPX)*nf1*nf2,cudaMemcpyDeviceToHost);
	for(int i=0; i<nf2; i++){
		for(int j=0; j<nf1; j++){
			printf("%.3g ",fw[i*nf1+j].real());
		}
		printf("\n");
	}
	free(fw);
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
	ier = curafft_deconv(plan);

	CPX *fk = (CPX *)malloc(sizeof(CPX)*N1*N2*N3);
	checkCudaErrors(cudaMemcpy(fk,plan->fk,sizeof(CUCPX)*N1*N2*N3, cudaMemcpyDeviceToHost));
	cudaEventRecord(cuda_end);

	cudaEventSynchronize(cuda_start);
	cudaEventSynchronize(cuda_end);
	cudaEventElapsedTime(&kernel_time, cuda_start, cuda_end);

	printf("Elapsed time: %.5g s\n",kernel_time/1000.0);
	
	// result printing
	// printf("final result printing...\n");
	// for(int i=0; i<N2; i++){
	for(int j=0; j<10; j++){
		printf("%.10lf ",fk[j].real());
	}
		printf("\n");
	// }
// 	int nt1 = (int)(0.37*N1), nt2 = (int)(0.26*N2), nt3 = (int) (0.13*N3);  // choose some mode index to check
// 	CPX Ft = CPX(0,0), J = IMA*(PCS)iflag;
// 	for (int j=0; j<M; ++j)
// 		Ft += c[j] * exp(J*(nt1*u[j]+nt2*v[j]+nt3*w[j]));   // crude direct
// 	int it = N1/2+nt1 + N1*(N2/2+nt2) + N1*N2*(N3/2+nt3);   // index in complex F as 1d array
// 	int N = N1*N2*N3;
// //	printf("[gpu   ] one mode: abs err in F[%ld,%ld,%ld] is %.3g\n",(int)nt1,
// //		(int)nt2, (int)nt3, (abs(Ft-fk[it])));
// 	printf("[gpu   ] one mode: rel err in F[%ld,%ld,%ld] is %.3g\n",(int)nt1,
// 		(int)nt2, (int)nt3, abs(Ft-fk[it])/infnorm(N,fk));
    // revise here wrong
	printf("ground truth printing...\n");
	CPX *truth = (CPX *) malloc(sizeof(CPX)*10);
	CPX Ft = CPX(0,0), J = IMA*(PCS)iflag;
	// for(int i=0; i<N2; i++){
		for(int j=0; j<10; j++){
			for (int k=0; k<M; ++k)
				Ft += c[k] * exp(J*((j-N1/2)*u[k]+(0-N2/2)*v[k]+(0-N3/2)*w[k]));   // crude direct
			truth[j] = Ft;
			printf("%.10lf ",Ft.real());
			Ft.real(0);
			Ft.imag(0);
		}
		printf("\n");
// }

	// double max=0;
	// double l2_max=0;
	// double fk_max = 0;
	// for(int i=0; i<M; i++){
	// 	if(abs(fk[i].real())>fk_max)fk_max = abs(fk[i].real());
	// }
	// printf("fk max %lf\n",fk_max);
	// for(int i=0; i<N1*N2; i++){
	// 	double temp = abs(truth[i].real()-fk[i].real());
	// 	if(temp>max) max = temp;
	// 	if(temp/fk_max > l2_max) l2_max = temp/fk_max;
	// }
	// printf("maximal abs error %.5g, maximal l2 error %.5g\n",max,l2_max);

	//free
	curafft_free(plan);
	free(fk);
	free(u);
	free(v);
    free(w);
	free(c);

	return ier;
}