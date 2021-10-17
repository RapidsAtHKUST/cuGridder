// Header for utils_fp.cpp, a little library of low-level array stuff.
// These are functions which depend on single/double precision.
// (rest of finufft defs and types are now in defs.h)

#if (!defined(UTILS_FP_H) && !defined(SINGLE)) || (!defined(UTILS_FPF_H) && defined(SINGLE))
// Make sure we only include once per precision (as in finufft_eitherprec.h).
#ifndef SINGLE
#define UTILS_FP_H
#else
#define UTILS_FPF_H
#endif

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

#include <complex> // C++ type complex
#include <cuComplex.h>
#include "datatype.h"

#undef EPSILON
#undef IMA
#undef FABS
#undef SET_NF_TYPE12

// Compile-flag choice of single or double (default) precision:
// (Note in the other codes, PCS is "double" or "float", CPX same but complex)
#ifdef SINGLE
// machine epsilon for rounding
#define EPSILON (float)6e-08
#define IMA std::complex<float>(0.0, 1.0)
#define FABS(x) fabs(x)
#define SET_NF_TYPE12 set_nf_type12f
#else
// machine epsilon for rounding
#define EPSILON (double)1.1e-16
#define IMA std::complex<double>(0.0, 1.0)
#define FABS(x) fabsf(x)
#define SET_NF_TYPE12 set_nf_type12
#endif

#define ARRAYWIDCEN_GROWFRAC 0.1

// ahb's low-level array helpers
PCS relerrtwonorm(int n, CPX *a, CPX *b);
PCS errtwonorm(int n, CPX *a, CPX *b);
PCS twonorm(int n, CPX *a);
PCS infnorm(int n, CPX *a);
void arrayrange(int n, PCS *a, PCS *lo, PCS *hi);
void indexedarrayrange(int n, int *i, PCS *a, PCS *lo, PCS *hi);
void arraywidcen(int n, PCS *a, PCS *w, PCS *c);

#endif // UTILS_FP_H
