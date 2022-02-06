#include "utils.h"
#include "utils_fp.h"


// ------------ complex array utils ---------------------------------

PCS relerrtwonorm(int n, CPX* a, CPX* b)
// ||a-b||_2 / ||a||_2
{
  PCS err = 0.0, nrm = 0.0;
  for (int m=0; m<n; ++m) {
    nrm += real(conj(a[m])*a[m]);
    CPX diff = a[m]-b[m];
    err += real(conj(diff)*diff);
  }
  return sqrt(err/nrm);
}

PCS errtwonorm(int n, CPX* a, CPX* b)
// ||a-b||_2
{
  PCS err = 0.0;   // compute error 2-norm
  for (int m=0; m<n; ++m) {
    CPX diff = a[m]-b[m];
    err += real(conj(diff)*diff);
  }
  return sqrt(err);
}

PCS twonorm(int n, CPX* a)
// ||a||_2
{
  PCS nrm = 0.0;
  for (int m=0; m<n; ++m)
    nrm += real(conj(a[m])*a[m]);
  return sqrt(nrm);
}

PCS infnorm(int n, CPX* a)
// ||a||_infty
{
  PCS nrm = 0.0;
  for (int m=0; m<n; ++m) {
    PCS aa = real(conj(a[m])*a[m]);
    if (aa>nrm) nrm = aa;
  }
  return sqrt(nrm);
}

void arrayrange(int n, PCS* a, PCS *lo, PCS *hi)
// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
// If n==0, lo and hi are not finite.
{
  *lo = INFINITY; *hi = -INFINITY;
  for (int m=0; m<n; ++m) {
    if (a[m]<*lo) *lo = a[m];
    if (a[m]>*hi) *hi = a[m];
  }
}

void indexedarrayrange(int n, int* i, PCS* a, PCS *lo, PCS *hi)
// With i a list of n indices, and a an array of length max(i), writes out
// min(a(i)) to lo and max(a(i)) to hi, so that all a(i) values lie in [lo,hi].
// This is not currently used in FINUFFT v1.2.
{
  *lo = INFINITY; *hi = -INFINITY;
  for (int m=0; m<n; ++m) {
    PCS A=a[i[m]];
    if (A<*lo) *lo = A;
    if (A>*hi) *hi = A;
  }
}

void arraywidcen(int n, PCS* a, PCS *w, PCS *c)
// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in defs.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
// If n==0, w and c are not finite.
{
  PCS lo,hi;
  arrayrange(n,a,&lo,&hi);
  *w = (hi-lo)/2;
  *c = (hi+lo)/2;
  if (std::abs(*c)<ARRAYWIDCEN_GROWFRAC*(*w)) {
    *w += std::abs(*c);
    *c = 0.0;
  }
}
