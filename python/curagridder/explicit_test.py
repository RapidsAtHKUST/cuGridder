from ducc0.wgridder import ms2dirty, dirty2ms
import numpy as np
import time
# import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal


# some functions are refered to NIFTY
def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


def explicit_gridder(uvw, freq, ms, nxdirty, nydirty, xpixsize, ypixsize, sign):
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
                     (x*uvw[row, 0] + y*uvw[row, 1] + sign*uvw[row, 2]*nm1))
            res += (ms[row, chan]*np.exp(2j*np.pi*phase)).real
    return res/n

def explicit_degridder(uvw, freq, dirty, xpixsize, ypixsize, nrow, nchan, sign):
    speedoflight = 299792458.
    nxdirty = dirty.shape[0]
    nydirty = dirty.shape[1]
    # ce shi
    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty]],
        indexing='ij')
    x *= xpixsize
    y *= ypixsize
    res = np.zeros((nrow,), dtype=np.complex128)

    dirty1 = np.zeros(dirty.shape,dtype=np.complex128)
    dirty1.real = dirty

    eps = x**2+y**2
    nm1 = -eps/(np.sqrt(1.-eps)+1.)
    n = nm1+1
    for row in range(nrow):
        for chan in range(nchan):
            phase = -(freq[chan]/speedoflight *
                     (x*uvw[row, 0] + y*uvw[row, 1] + sign*uvw[row, 2]*nm1))
            res[row] = np.sum((dirty1/n*np.exp(2j*np.pi*phase)))
    return res

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
    uvw = (np.random.rand(nrow, 3)-0.5)/(f0/speedoflight)/xpixsize
    ms = np.random.rand(nrow, nchan)-0.5 + 1j*(np.random.rand(nrow, nchan)-0.5)
    dirty = np.random.rand(nxdirty, nydirty)-0.5
    wgt = np.random.rand(nrow, nchan)
    
    print("begin")
    start = time.time()
    dirty2 = ms2dirty(uvw,freq, ms, None, nxdirty, nydirty, xpixsize, ypixsize, 0, 0, epsilon, True, 32)
    end = time.time()
    print("The elapsed time {} (sec)".format(end-start))
    print("Execution finished")
    dirty2 = np.reshape(dirty2,[nxdirty,nydirty])
    ms2 = np.zeros((nrow,1),dtype=np.complex128)
    start = time.time()
    ms2 = dirty2ms(uvw,freq, dirty, None, xpixsize, ypixsize, 0, 0, epsilon, True, 32)
    end = time.time()
    print("The elapsed time {} (sec)".format(end-start))
    print("Execution finished")

    # truth_ms = explicit_degridder(uvw, freq, dirty, xpixsize, ypixsize, nrow, nchan, -1)
    # print("L2 error between explicit degridding and CURIG:",
    #           _l2error(truth_ms.real, np.squeeze(ms2.real)))

    # ms2 = np.reshape(ms2,[nrow,1])
    print("\nadjointness testing....")
    print(np.vdot(ms, ms2).real)
    print(np.vdot(dirty2, dirty).real)
    # assert_allclose(np.vdot(ms, ms2).real, np.vdot(dirty2, dirty).real, rtol=1e-12)

    # if nrow<1e4:
    #     print("Vertification begin")
    #     truth = explicit_gridder(uvw, freq, ms, nxdirty, nydirty, xpixsize, ypixsize, -1)
    #     print("L2 error between explicit gridding and CURIG:",
    #           _l2error(truth, dirty2))

   

# the first test will execute 2 times to warp up the GPU
# for i in range(10):
# test_against_wdft(100000, 1, 2048, 2048, 2, 1e-12)
# test_against_wdft(1000000, 1, 2048, 2048, 2, 1e-12)
# test_against_wdft(10000000, 1, 2048, 2048, 2, 1e-12)
# test_against_wdft(100000000, 1, 2048, 2048, 2, 1e-12)

test_against_wdft(100000, 1, 4096, 4096, 10, 1e-12)
test_against_wdft(1000000, 1, 4096, 4096, 10, 1e-12)
test_against_wdft(10000000, 1, 4096, 4096, 10, 1e-12)
test_against_wdft(100000000, 1, 4096, 4096, 10, 1e-12)
