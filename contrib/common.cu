#include <math.h>
#include <stdio.h>
#include <vector>
#include "common.h"
#include "curafft_plan.h"
#ifdef __cplusplus
extern "C"
{
#include "legendre_rule_fast.h"
}
#else
#include "legendre_rule_fast.h"
#endif




PCS evaluate_kernel(PCS x, const conv_opts &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 2/17/17 */
{
  if (abs(x)>=opts.ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return exp(opts.ES_beta * sqrt(1.0 - opts.ES_c*x*x));
}

void onedim_fseries_kernel_seq(int nf, PCS *fwkerhalf, conv_opts opts)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid. Note that this is also the Fourier transform of the non-periodized
  kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  overall prefactor of 1/h, which is needed anyway for the correction, and
  arises because the quadrature weights are scaled for grid units not x units.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
              divided by h = 2pi/n.
              (should be allocated for at least nf/2+1 PCSs)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.

  Barnett 2/7/17. openmp (since slow vs fftw in 1D large-N case) 3/3/18
 */
{
  PCS J2 = opts.kw / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 3.0 * J2); // not sure why so large? cannot exceed MAX_NQUAD
  PCS f[MAX_NQUAD];
  double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];
  legendre_compute_glr(2 * q, z, w); // only half the nodes used, eg on (0,1)
  std::complex<double> a[MAX_NQUAD];
  for (int n = 0; n < q; ++n)
  {                                                            // set up nodes z_n and vals f_n
    z[n] *= J2;                                                // rescale nodes
    f[n] = J2 * (PCS)w[n] * evaluate_kernel((PCS)z[n], opts);  // vals & quadr wei
    //a[n] = exp(2 * PI * IMA * (PCS)(nf / 2 - z[n]) / (PCS)nf); // phase winding rates
  }
  printf("z printing...\n");
  for(int i=0; i<2*q; i++){
    printf("%.5lf ",z[i]/J2);
  }
  printf("\n");
  for(int i =0; i<nf/2+1; i++){
    PCS x = 0.0; // accumulator for answer at this j
        for (int n = 0; n < q; ++n)
        {
          x += f[n] * 2 * cos(i * 2 * PI * (PCS)(nf/2-z[n]) / (PCS)nf); // include the negative freq
        }
        fwkerhalf[i] = x;
  }
}

void onedim_fseries_kernel(int nf, PCS *fwkerhalf, conv_opts opts)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid. Note that this is also the Fourier transform of the non-periodized
  kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  overall prefactor of 1/h, which is needed anyway for the correction, and
  arises because the quadrature weights are scaled for grid units not x units.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
              divided by h = 2pi/n.
              (should be allocated for at least nf/2+1 PCSs)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.

  Barnett 2/7/17. openmp (since slow vs fftw in 1D large-N case) 3/3/18
 */
{
  PCS J2 = opts.kw/2.0;            // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q=(int)(2 + 3.0*J2);  // not sure why so large? cannot exceed MAX_NQUAD
  PCS f[MAX_NQUAD]; double z[2*MAX_NQUAD],w[2*MAX_NQUAD];
  legendre_compute_glr(2*q,z,w);        // only half the nodes used, eg on (0,1)
  std::complex<double> a[MAX_NQUAD];
  for (int n=0;n<q;++n) {               // set up nodes z_n and vals f_n
    z[n] *= J2;                         // rescale nodes
    f[n] = J2*(PCS)w[n] * evaluate_kernel((PCS)z[n], opts); // vals & quadr wei
    a[n] = exp(2*PI*IMA*(PCS)(nf/2-z[n])/(PCS)nf);  // phase winding rates
  }
  int nout=nf/2+1;                   // how many values we're writing to
  int nt = MIN(nout,MY_OMP_GET_MAX_THREADS());  // how many chunks
  std::vector<int> brk(nt+1);        // start indices for each thread
  for (int t=0; t<=nt; ++t)             // split nout mode indices btw threads
    brk[t] = (int)(0.5 + nout*t/(double)nt);
#pragma omp parallel
  {
    int t = MY_OMP_GET_THREAD_NUM();
    if (t<nt) {                         // could be nt < actual # threads
      std::complex<double> aj[MAX_NQUAD];           // phase rotator for this thread
      for (int n=0;n<q;++n)
	aj[n] = pow(a[n],(PCS)brk[t]);       // init phase factors for chunk
      for (int j=brk[t];j<brk[t+1];++j) {       // loop along output array
	PCS x = 0.0;                       // accumulator for answer at this j
	for (int n=0;n<q;++n) {
	  x += f[n] * 2*real(aj[n]);       // include the negative freq
	  aj[n] *= a[n];                   // wind the phases
	}
	fwkerhalf[j] = x;
      }
    }
  }
}
void set_nhg_type3(PCS S, PCS X, conv_opts spopts,
		     int &nf, PCS &h, PCS &gam)
/* sets nf, h (upsampled grid spacing), and gamma (x_j rescaling factor),
   for type 3 only.
   Inputs:
   X and S are the xj and sk interval half-widths respectively.
   opts and spopts are the NUFFT and spreader opts strucs, respectively.
   Outputs:
   nf is the size of upsampled grid for a given single dimension.
   h is the grid spacing = 2pi/nf
   gam is the x rescale factor, ie x'_j = x_j/gam  (modulo shifts).
   Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
   New logic 6/12/17
*/
{
  int nss = spopts.kw + 1;      // since ns may be odd
  PCS Xsafe=X, Ssafe=S;              // may be tweaked locally
  if (X==0.0)                        // logic ensures XS>=1, handle X=0 a/o S=0
    if (S==0.0) {
      Xsafe=1.0;
      Ssafe=1.0;
    } else Xsafe = std::max(Xsafe, 1/S);
  else
    Ssafe = std::max(Ssafe, 1/X);
  // use the safe X and S...
  PCS nfd = 2.0*spopts.upsampfac*Ssafe*Xsafe/PI + nss;
  if (!isfinite(nfd)) nfd=0.0;                // use FLT to catch inf
  nf = (int)nfd;
  //printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  // catch too small nf, and nan or +-inf, otherwise spread fails...
  if (nf<2*spopts.kw) nf=2*spopts.kw;
  if (nf<MAX_NF)                             // otherwise will fail anyway
  {
    int n = nf;
    if (n<=2) n = 2;
    else{
    if (n%2 == 1) n+=1;   // even
    int nplus = n-2;   // to cancel out the +=2 at start of loop
    int numdiv = 2;    // a dummy that is >1
    while (numdiv>1) {
    nplus += 2;         // stays even
    numdiv = nplus;
    while (numdiv%2 == 0) numdiv /= 2;  // remove all factors of 2,3,5...
    while (numdiv%3 == 0) numdiv /= 3;
    while (numdiv%5 == 0) numdiv /= 5;
    }
    n = nplus;
    }
    nf = n;
  }
  h = 2*PI / nf;                            // upsampled grid spacing
  gam = (PCS)nf / (2.0*spopts.upsampfac*Ssafe);  // x scale fac to x'
}