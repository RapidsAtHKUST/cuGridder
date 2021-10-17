#ifndef __DATATYPE_H__
#define __DATATYPE_H__

/* ------------ data type definitions ----------------- */
#include <cuComplex.h>
#include <cuda.h>
#include <math.h>
#include <complex>
// using namespace std::complex_literals;
//complex not thrust/complex
#define COMPLEX(X) std::complex<X>

#undef PCS
#undef CPX

//define precision
#ifdef SINGLE
#define PCS float
#define CUCPX cuFloatComplex
#define CUFFT_TYPE CUFFT_C2C
#define CUFFT_EXEC cufftExecC2C
#else
#define PCS double
#define CUCPX cuDoubleComplex
#define CUFFT_TYPE CUFFT_Z2Z
#define CUFFT_EXEC cufftExecZ2Z
#endif

#define CPX COMPLEX(PCS)

#define INT_M int3

#endif
