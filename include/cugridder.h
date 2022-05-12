#ifndef __CUGRIDDER_H__
#define __CUGRIDDER_H__

#include "curafft_plan.h"
#include "ragridder_plan.h"
#include "utils.h"

#ifdef MS2DIRTY_2
#undef MS2DIRTY_2
#endif
#ifdef MS2DIRTY_1
#undef MS2DIRTY_1
#endif
#ifdef DIRTY2MS_1
#undef DIRTY2MS_1
#endif
#ifdef DIRTY2MS_2
#undef DIRTY2MS_2
#endif

#ifdef SINGLE
#define MS2DIRTY_2 ms2dirtyf_2
#define MS2DIRTY_1 ms2dirtyf_1
#define DIRTY2MS_1 dirty2msf_1
#define DIRTY2MS_2 dirty2msf_2
#else
#define MS2DIRTY_2 ms2dirty_2
#define MS2DIRTY_1 ms2dirty_1
#define DIRTY2MS_1 dirty2ms_1
#define DIRTY2MS_2 dirty2ms_2
#endif

int gridder_setting(int N1, int N2, int method, int kerevalmeth, int w_term_method, PCS tol, int direction, double sigma, int iflag,
                    int batchsize, int M, int channel, PCS fov, visibility *pointer_v, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, CURAFFT_PLAN *plan, ragridder_plan *gridder_plan);
int gridder_execution(CURAFFT_PLAN *plan, ragridder_plan *gridder_plan);
int gridder_destroy(CURAFFT_PLAN *plan, ragridder_plan *gridder_plan);
extern "C"
{
    int MS2DIRTY_2(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
                   CPX *vis, PCS *wgt, CPX *dirty, PCS epsilon, PCS sigma, int sign);
    int MS2DIRTY_1(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
                   CPX *vis, CPX *dirty, PCS epsilon, PCS sigma, int sign);
    int DIRTY2MS_1(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
                   CPX *vis, CPX *dirty, PCS epsilon, PCS sigma, int sign);
    int DIRTY2MS_2(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
                   CPX *vis, PCS *wgt, CPX *dirty, PCS epsilon, PCS sigma, int sign);
}
#endif