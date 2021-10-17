from curig import ms2dirty, dirty2ms
import numpy as np
import time
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal

pmp = pytest.mark.parametrize

# some functions are refered to NIFTY
# test this content by - pytest python/curagridder/test_wst.py   (test after curig installed)
def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


def explicit_gridder(uvw, freq, ms, nxdirty, nydirty, xpixsize, ypixsize):
    speedoflight = 299792458.

    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty]],
        indexing='ij')
    x *= xpixsize
    y *= ypixsize
    res = np.zeros((nxdirty, nydirty))

    eps = x**2+y**2
    nm1 = -eps/(np.sqrt(1.-eps)+1.)
    n = nm1+1
    for row in range(ms.shape[0]):
        for chan in range(ms.shape[1]):
            phase = (freq[chan]/speedoflight *
                     (x*uvw[row, 0] + y*uvw[row, 1] - uvw[row, 2]*nm1))
            res += (ms[row, chan]*np.exp(2j*np.pi*phase)).real
    return res/n



@pmp("nrow", (2, 27, 100))
@pmp("nchan", (1, ))
@pmp("nxdirty", (32, 64, 16))
@pmp("nydirty", (128, 250, 64))
@pmp("fov",(1, 10, 20))
@pmp("epsilon", (2e-1, 5e-3, 5e-5, 5e-7, 5e-12))
@pmp("singleprec", (False, True))
@pmp("use_wgt", (False, True))

def test_against_wdft(nrow, nchan, nxdirty, nydirty, fov, epsilon, singleprec, use_wgt):
    print("\n\nTesting imaging with {} rows and {} "
          "frequency channels".format(nrow, nchan))
    print("Dirty image has {}x{} pixels, "
          "FOV={} degrees".format(nxdirty, nydirty, fov))
    print("Requested accuracy: {}".format(epsilon))

    if singleprec and epsilon < 5e-6:
            return

    xpixsize = fov*np.pi/180/nxdirty
    ypixsize = fov*np.pi/180/nydirty

    speedoflight = 299792458.
    np.random.seed(42)
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (np.random.rand(nrow, 3)-0.5)/(f0/speedoflight)
    ms = np.random.rand(nrow, nchan)-0.5 + 1j*(np.random.rand(nrow, nchan)-0.5)
    dirty = np.random.rand(nxdirty, nydirty)-0.5
    dirty2 = np.zeros((nxdirty,nydirty),dtype=np.float64)
    wgt = np.random.rand(nrow, nchan) if use_wgt else None
    if singleprec:
        ms = ms.astype("c8")
        dirty = dirty.astype("f4")
        if wgt is not None:
            wgt = wgt.astype("f4")


    print("begin")
    start = time.time()
    dirty2 = ms2dirty(uvw, freq, ms, wgt, nxdirty, nydirty, xpixsize, ypixsize, 0, 0, epsilon, True, 4).astype("f8")
    end = time.time()
    print("The elapsed time {} (sec)".format(end-start))
    print("Execution finished")
    
    ms2 = np.zeros((nrow,1),dtype=np.complex128)
    ms2 = dirty2ms(uvw,freq, dirty, wgt, xpixsize, ypixsize, 0, 0, epsilon, True, 4).astype("c16")

    tol = 5e-5 if singleprec else 1e-12
    # ms2 = np.reshape(ms2,[nrow,1])
    print("\nadjointness testing....")
    print(np.vdot(ms, ms2).real)
    print(np.vdot(dirty2, dirty).real)
    assert_allclose(np.vdot(ms, ms2).real, np.vdot(dirty2, dirty), rtol=tol)
    
    if nrow<1e4:
        print("Vertification begin")
        truth = explicit_gridder(uvw, freq, ms, nxdirty, nydirty, xpixsize, ypixsize)
        print("L2 error between explicit transform and CURIG:",
              _l2error(truth, dirty2))

   

# the first test will execute 2 times to warp up the GPU
# for i in range(10):
# test_against_wdft(1000, 1, 512, 512, 2, 1e-12)
# test_against_wdft(1000, 1, 512, 512, 2, 1e-12)


# test_against_wdft(10000, 1, 512, 512, 60, 1e-12)

# test_against_wdft(10000, 1, 1024, 1024, 2, 1e-12)
# test_against_wdft(100000000, 1, 1024, 1024, 2, 1e-12)
# test_against_wdft(100000000, 1, 2048, 2048, 2, 1e-12)
# test_against_wdft(100000000, 1, 4096, 4096, 2, 1e-12)