#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>

#include "precomp.h"
#include "utils.h"
#include "ragridder_plan.h"

int main(int argc, char *argv[])
{
    /*
        explicit gridder test
        nrow - number of rows
        nchan - number of channels
        nxdirty, nydirty - image size (height, width)
        fov - field of view
    */
	if (argc < 5)
	{
		fprintf(stderr,
				"Usage: Explicit gridder\n"
				"Arguments:\n"
				"  nrow: The number of rows.\n"
				"  nxdirty, nydirty : image size.\n"
				"  fov: Field of view.\n"
				"  nchan: number of chanels (default 1)\n"
				);
		return 1;
	}
    int nrow;
    int nxdirty, nydirty;
	int nchan = 1;
	PCS fov;

	double inp;

	sscanf(argv[1], "%d", &nrow);
	sscanf(argv[2], "%d", &nxdirty);
	sscanf(argv[3], "%d", &nydirty);
	sscanf(argv[4], "%lf", &inp);
	fov = inp;

	if (argc > 5)
	{
		sscanf(argv[5], "%d", &nchan);
	}


	// degree per pixel (unit radius)
	PCS deg_per_pixelx = fov / 180.0 * PI / (PCS)nxdirty;
	PCS deg_per_pixely = fov / 180.0 * PI / (PCS)nydirty;
	// chanel setting
	PCS f0 = 1e9;
	PCS *freq = (PCS *)malloc(sizeof(PCS) * nchan);
	for (int i = 0; i < nchan; i++)
	{
		freq[i] = f0 + i / (double)nchan * f0; //!
	}
	//improved WS stacking 1,
	//gpu_method == 0, nupts driven

	//N1 = 5; N2 = 5; M = 25; //for correctness checking
	//int ier;
	PCS *u, *v, *w;
	CPX *vis;
	u = (PCS *)malloc(nrow * sizeof(PCS)); //Allocates page-locked memory on the host.
	v = (PCS *)malloc(nrow * sizeof(PCS));
	w = (PCS *)malloc(nrow * sizeof(PCS));
	vis = (CPX *)malloc(nrow * sizeof(CPX));

	// generating data
	for (int i = 0; i < nrow; i++)
	{
		u[i] = i ;
		v[i] = i ;
		w[i] = i ;
		vis[i].real(i); // nrow vis per channel, weight?
		vis[i].imag(i);
		// wgt[i] = 1;
	}
    // gridder plan setting
    ragridder_plan * plan = (ragridder_plan *) malloc (sizeof(ragridder_plan));
    memset(plan, 0, sizeof(ragridder_plan));

    plan->channel = nchan;
    plan->fov = fov;
    plan->height = nxdirty;
    plan->width = nydirty;
    plan->nrow = nrow;
    plan->pixelsize_x = deg_per_pixelx;
    plan->pixelsize_y = deg_per_pixely;
    plan->speedoflight = SPEEDOFLIGHT;
    plan->kv.pirange = 0;
    plan->kv.frequency = freq;
    plan->kv.u = u;
    plan->kv.v = v;
    plan->kv.w = w;
    plan->kv.vis = vis;


    plan->dirty_image = (CPX *)malloc(sizeof(CPX)*nxdirty*nydirty*nchan); //

    explicit_gridder_invoker(plan,0);

    // result printing
	printf("result printing...\n");
    for(int i=0; i<nxdirty; i++){
        for(int j=0; j<nydirty; j++){
            printf("%lf ",plan->dirty_image[i*nydirty+j].real());
        }
        printf("\n");
    }

	PCS *truth = (PCS *) malloc(sizeof(PCS)*nxdirty*nydirty);
	// add ground truth result printing and error printing +++
	printf("ground truth printing...\n");
	PCS pi_ratio = 1;
	if(!plan->kv.pirange)pi_ratio = 2 * PI;
	for(int i=0; i<nxdirty; i++){
		for(int j=0; j<nydirty; j++){
			CPX temp(0.0,0.0);
			PCS n_lm = sqrt(1.0-pow(plan->pixelsize_x*(i-nxdirty/2),2)-pow(plan->pixelsize_y*(j-nydirty/2),2));
			
			for(int k=0; k<nrow; k++){

				PCS phase = f0/SPEEDOFLIGHT*(u[k]*pi_ratio*plan->pixelsize_x*(i-nxdirty/2)+v[k]*pi_ratio*plan->pixelsize_y*(j-nydirty/2)+w[k]*pi_ratio*(n_lm-1));
				temp += vis[k]*exp(phase*IMA);
			}
			printf("%lf ",temp.real()/(n_lm));
			truth[i*nydirty+j] = temp.real()/(n_lm);
		}
		printf("\n");
	}
	
	double max = 0.0;
	for(int i=0; i<nxdirty; i++){
		for(int j=0; j<nydirty; j++){
			double temp = abs(truth[i]-plan->dirty_image[i].real());
			if(temp>max) max =temp;
		}
	}
	printf("maximal absolute error %.5g\n",max);
	// double max=0;
	// double l2_max=0;

	// for(int i=0; i<nydirty*nxdirty; i++){
	// 	double temp = abs(truth[i]-plan->dirty_image[i].real());
	// 	if(temp>max) max = temp;
	// 	if(temp/plan->dirty_image[i].real() > l2_max) l2_max = temp/plan->dirty_image[i].real();
	// }
	// printf("max abs error %.10lf, max l2 error %.10lf\n",max,l2_max);
    return 0;
}