#ifndef __CUFT_H__
#define __CUFT_H__
		     

#include "curafft_plan.h"
int setup_plan(int nf1, int nf2, int nf3, int M, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, CURAFFT_PLAN *plan);
void curadft_invoker(CURAFFT_PLAN *plan, PCS xpixelsize, PCS ypixelsize);
void curadft_partial_invoker(CURAFFT_PLAN *plan, PCS xpixelsize, PCS ypixelsize, int plane_id);
int cunufft_setting(int N1, int N2, int N3, int M, int kerevalmeth, int method, int direction, PCS tol,  PCS sigma, int type, int dim,
                        PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, CURAFFT_PLAN *plan);
void pre_stage_invoker(PCS *i_center, PCS *o_center, PCS *gamma, PCS *h, PCS *d_u, PCS *d_v, PCS *d_w, PCS *d_x, PCS *d_y, PCS *d_z, 
                        CUCPX *d_c, int M, int N1, int N2, int N3, int flag);
void pre_stage_1_invoker(PCS *o_center, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, int M, int flag);
void pre_stage_1_invoker(PCS i_center, PCS *d_z, CUCPX *d_fk, int N1, int N2, PCS xpixelsize, PCS ypixelsize, int flag);
void pre_stage_2_invoker(PCS *i_center, PCS *o_center, PCS *gamma, PCS *h, PCS *d_u, PCS *d_v, PCS *d_w, PCS *d_x, PCS *d_y, PCS *d_z,
                        CUCPX *d_c, int M, int N1, int N2, int N3);
int cura_prestage(CURAFFT_PLAN *plan);
int cura_cufft(CURAFFT_PLAN *plan);
int cufft_plan_setting(CURAFFT_PLAN *plan);
void taylor_coefficient_setting(CURAFFT_PLAN *plan);
#endif