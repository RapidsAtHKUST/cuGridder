#ifndef __RA_EXEC_H__
#define __RA_EXEC_H__
#include "curafft_plan.h"
#include "ragridder_plan.h"

int exec_vis2dirty(CURAFFT_PLAN *plan, ragridder_plan *gridder_plan);
int exec_dirty2vis(CURAFFT_PLAN *plan, ragridder_plan *gridder_plan);
#endif