#ifndef __CONV_INTERP_INVOKER_H__
#define __CONV_INTERP_INVOKER_H__


#include "curafft_plan.h"
#include "conv.h"

#define CONV_THREAD_NUM 32

int setup_conv_opts(conv_opts &c_opts, PCS eps, PCS upsampfac, int pirange, int direction, int kerevalmeth);//cautious the &
int bin_mapping(CURAFFT_PLAN *plan, PCS *uvw);
int get_num_cells(PCS ms, conv_opts copts);
int curafft_conv(CURAFFT_PLAN *plan);
int curafft_free(CURAFFT_PLAN *plan);
int curafft_interp(CURAFFT_PLAN * plan);
int part_bin_mapping_pre(CURAFFT_PLAN *plan, int *temp_station, int &initial);
int part_bin_mapping(CURAFFT_PLAN *plan, PCS *d_u_out, PCS *d_v_out, PCS *d_w_out, CUCPX *d_c_out, unsigned long long int histo_count_size, int cube_id, int &initial);
int curaff_partial_conv(CURAFFT_PLAN *plan, int init_shift, int up_shift, int c_shift, int down_shift);
int curaff_partial_interp(CURAFFT_PLAN *plan, int start_pos, int end_pos, int num_beg, int plane_id, int cube_size);
#endif
