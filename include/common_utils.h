#ifndef __COMMON_UTILS_H__
#define __COMMON_UTILS_H__

#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
//#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "datatype.h"


void GPU_info();
void show_mem_usage();
int next235beven(int n, int b);
#endif