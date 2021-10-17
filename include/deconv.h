#ifndef __DECONV_H__
#define __DECONV_H__

#include "curafft_plan.h"
#ifdef __cplusplus
extern "C"
{
#include "../../contrib/legendre_rule_fast.h"
}
#else
#include "../../contrib/legendre_rule_fast.h"
#endif
int fourier_series_appro_invoker(PCS *fseries, conv_opts opts, int N);
int fourier_series_appro_invoker(PCS *fseries, PCS *k, conv_opts opts, int N, int nf);
int fourier_series_appro_invoker(PCS *fseries, PCS *k, conv_opts opts, int N);
int curafft_deconv(CURAFFT_PLAN *plan);

// below this line, all contents ares pecified for radio astronomy
int curadft_w_deconv(CURAFFT_PLAN *plan, PCS xpixelsize, PCS ypixelsize);
#endif