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
void counting_hive_invoker(int *hive_count, int *histo_count, unsigned long int hive_count_size, int hivesize);
void prefix_scan(int *d_arr, int *d_res, unsigned long int n, int flag);
#endif