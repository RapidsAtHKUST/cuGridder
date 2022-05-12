#ifndef __CURAFFT_OPTS_H__
#define __CURAFFT_OPTS_H__

struct curafft_opts
{
	/*
		upsampfac - upsampling ratio sigma, only 2.0 (standard) is implemented
		gpu_device_id - for choosing GPU
		gpu_sort - 1 using sort, 0 not

		add content explaination
	*/

	double upsampfac; // upsampling ratio sigma, only 2.0 (standard) is implemented

	/* multi-gpu support */
	int gpu_device_id;

	/* For GM_sort method*/
	int gpu_sort; // 1 using sort.
	int gpu_binsizex;
	int gpu_binsizey;
	int gpu_binsizez;
	int gpu_kerevalmeth;	// 0: direct exp(sqrt()), 1: Horner ppval default 0
	int gpu_conv_only;		// 0: NUFFT, 1: conv only
	int gpu_gridder_method; // 0: nupt, 1: sorted_nupt, 2:partical conv

	curafft_opts()
	{
		gpu_device_id = 0;
		upsampfac = 2.0;
		gpu_sort = 0;
		gpu_binsizex = -1;
		gpu_binsizey = -1;
		gpu_binsizez = -1;
		gpu_kerevalmeth = 0;
		gpu_conv_only = 0;
	};
};

#endif