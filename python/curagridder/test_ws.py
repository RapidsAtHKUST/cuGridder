from curig import vis2dirty, dirty2vis
import numpy as np
import time
# import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal


# some functions are refered to NIFTY, the image size should be even, due to the ss/2 is not a integer
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
                     (x*uvw[row, 0] + y*uvw[row, 1] + uvw[row, 2]*nm1))
            res += (ms[row, chan]*np.exp(2j*np.pi*phase)).real
    return res/n


def test_against_wdft(nrow, nchan, nxdirty, nydirty, fov, epsilon):
    print("\n\nTesting imaging with {} rows and {} "
          "frequency channels".format(nrow, nchan))
    print("Dirty image has {}x{} pixels, "
          "FOV={} degrees".format(nxdirty, nydirty, fov))
    print("Requested accuracy: {}".format(epsilon))

    xpixsize = fov*np.pi/180/nxdirty
    ypixsize = fov*np.pi/180/nydirty

    speedoflight = 299792458.
    np.random.seed(42)
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (np.random.rand(nrow, 3)-0.5)/(f0/speedoflight)
    ms = np.random.rand(nrow, nchan)-0.5 + 1j*(np.random.rand(nrow, nchan)-0.5)
    dirty = np.random.rand(nxdirty, nydirty)-0.5 + 0j
    dirty2 = np.zeros((nxdirty,nydirty),dtype=np.complex128)
    
    print("begin")
    start = time.time()
    dirty2 = vis2dirty(uvw,freq, ms, None, dirty2, fov, epsilon,2)
    end = time.time()
    print("The elapsed time {} (sec)".format(end-start))
    print("Execution finished")
    dirty2 = np.reshape(dirty2,[nxdirty,nydirty])
    ms2 = np.zeros((nrow,1),dtype=np.complex128)
    ms2 = dirty2vis(uvw,freq, ms2, None, dirty, fov, epsilon,2)

    # ms2 = np.reshape(ms2,[nrow,1])
    print("\nadjointness testing....")
    print(np.vdot(ms, ms2).real)
    print(np.vdot(dirty2, dirty).real)
    assert_allclose(np.vdot(ms, ms2).real, np.vdot(dirty2, dirty).real, rtol=1e-12)

    if nrow<1e4:
        print("Vertification begin")
        truth = explicit_gridder(uvw, freq, ms, nxdirty, nydirty, xpixsize, ypixsize)
        print("L2 error between explicit transform and CURIG:",
              _l2error(truth, dirty2.real))

   

# the first test will execute 2 times to warp up the GPU
# for i in range(10):
test_against_wdft(1000, 1, 512, 512, 2, 1e-12)
test_against_wdft(1000, 1, 512, 512, 2, 1e-12)


test_against_wdft(10000, 1, 512, 512, 60, 1e-12)

# test_against_wdft(10000, 1, 1024, 1024, 2, 1e-12)
# test_against_wdft(100000000, 1, 1024, 1024, 2, 1e-12)
# test_against_wdft(100000000, 1, 2048, 2048, 2, 1e-12)
# test_against_wdft(100000000, 1, 4096, 4096, 2, 1e-12)


