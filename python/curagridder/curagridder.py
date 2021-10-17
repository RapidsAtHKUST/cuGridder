#import pycuda.autoinit # NOQA:401
#import pycuda.gpuarray as gpuarray

import ctypes
import os
import warnings

import numpy as np
from ctypes import c_double
from ctypes import c_int
from ctypes import c_float
from ctypes import c_void_p

c_int_p = ctypes.POINTER(c_int)
c_float_p = ctypes.POINTER(c_float)
c_double_p = ctypes.POINTER(c_double)

# TODO: See if there is a way to improve this so it is less hacky.
lib = None
# Try to load a local library directly.
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"libcurafft.so")
try:
    lib = ctypes.cdll.LoadLibrary(lib_path)
except Exception:
    raise RuntimeError('Failed to find curagridder library')


def _get_ctypes(dtype):
    """
    Checks dtype is float32 or float64.
    Returns floating point and floating point pointer.
    Y. Shih, G. Wright, J. Andén, J. Blaschke, A. H. Barnett (2021). cuFINUFFT
    """

    if dtype == np.float64:
        REAL_t = c_double
    elif dtype == np.float32:
        REAL_t = c_float
    else:
        raise TypeError("Expected np.float32 or np.float64.")

    REAL_ptr = ctypes.POINTER(REAL_t)

    return REAL_t, REAL_ptr

#double
ms2dirty_1 = lib.ms2dirty_1
# the last two parameters have default value
ms2dirty_1.argtypes = [c_int, c_int, c_int, c_double, c_double, np.ctypeslib.ndpointer(np.double, flags='C'),
                     np.ctypeslib.ndpointer(np.complex128, flags='C'), np.ctypeslib.ndpointer(np.complex128, flags='C'), c_double, c_double, c_int] 
ms2dirty_1.restype = c_int

ms2dirty_2 = lib.ms2dirty_2
ms2dirty_2.argtypes = [c_int, c_int, c_int, c_double, c_double, np.ctypeslib.ndpointer(np.double, flags='C'),
                     np.ctypeslib.ndpointer(np.complex128, flags='C'), np.ctypeslib.ndpointer(np.double, flags='C'), np.ctypeslib.ndpointer(np.complex128, flags='C'), c_double, c_double, c_int] 
ms2dirty_2.restype = c_int

dirty2ms_1 = lib.dirty2ms_1
# the last two parameters have default value
dirty2ms_1.argtypes = [c_int, c_int, c_int, c_double, c_double, np.ctypeslib.ndpointer(np.double, flags='C'),
                     np.ctypeslib.ndpointer(np.complex128, flags='C'), np.ctypeslib.ndpointer(np.complex128, flags='C'), c_double, c_double, c_int] 
dirty2ms_1.restype = c_int

dirty2ms_2 = lib.dirty2ms_2
dirty2ms_2.argtypes = [c_int, c_int, c_int, c_double, c_double, np.ctypeslib.ndpointer(np.double, flags='C'),
                     np.ctypeslib.ndpointer(np.complex128, flags='C'), np.ctypeslib.ndpointer(np.double, flags='C'), np.ctypeslib.ndpointer(np.complex128, flags='C'), c_double, c_double, c_int] 
dirty2ms_2.restype = c_int

#float
ms2dirtyf_1 = lib.ms2dirtyf_1
# the last two parameters have default value
ms2dirtyf_1.argtypes = [c_int, c_int, c_int, c_float, c_float, np.ctypeslib.ndpointer(np.float32, flags='C'),
                     np.ctypeslib.ndpointer(np.complex64, flags='C'), np.ctypeslib.ndpointer(np.complex64, flags='C'), c_float, c_float, c_int] 
ms2dirtyf_1.restype = c_int

ms2dirtyf_2 = lib.ms2dirtyf_2
ms2dirtyf_2.argtypes = [c_int, c_int, c_int, c_double, c_double, np.ctypeslib.ndpointer(np.float32, flags='C'),
                     np.ctypeslib.ndpointer(np.complex64, flags='C'), np.ctypeslib.ndpointer(np.float32, flags='C'), np.ctypeslib.ndpointer(np.complex64, flags='C'), c_float, c_float, c_int] 
ms2dirtyf_2.restype = c_int

dirty2msf_1 = lib.dirty2msf_1
# the last two parameters have default value
dirty2msf_1.argtypes = [c_int, c_int, c_int, c_float, c_float, np.ctypeslib.ndpointer(np.float32, flags='C'),
                     np.ctypeslib.ndpointer(np.complex64, flags='C'), np.ctypeslib.ndpointer(np.complex64, flags='C'), c_float, c_float, c_int] 
dirty2msf_1.restype = c_int

dirty2msf_2 = lib.dirty2msf_2
dirty2msf_2.argtypes = [c_int, c_int, c_int, c_float, c_float, np.ctypeslib.ndpointer(np.float32, flags='C'),
                     np.ctypeslib.ndpointer(np.complex64, flags='C'), np.ctypeslib.ndpointer(np.float32, flags='C'), np.ctypeslib.ndpointer(np.complex64, flags='C'), c_float, c_float, c_int] 
dirty2msf_2.restype = c_int

def vis2dirty(uvw, freq, ms, wgt, dirty, fov, epsilon=1e-6,sigma=2):
    """
    Generate an image from visibility by non-uniform fourier transform
    Arguments:
        uvw - 3D coordinates, numpy array, shape - (nrow,3)
        freq - frequencies
        vis - visibility, shape - (nrow,)
        wgt - weight
        nxdirty, nydirty - image size
        fov - field of view
        epsilon - tolerance of relative error (expect, default 1e-6)
        sigma - upsampling factor for grid (default 2)
    Return:
        dirty image - shape-[nxdirty,nydirty]
    """
    nrow = uvw.shape[0]
    nxdirty = dirty.shape[0]
    nydirty = dirty.shape[1]
    sign = 1
    # u = np.ctypeslib.as_ctypes(uvw[:,0])
    # v = np.ctypeslib.as_ctypes(uvw[:,1])
    # w = np.ctypeslib.as_ctypes(uvw[:,2])
    if(wgt is None):
        ms2dirty_1(nrow,nxdirty,nydirty,fov,freq[0],uvw
            ,ms,dirty,epsilon,sigma,sign)
    else:
        ms2dirty_2(nrow,nxdirty,nydirty,fov,freq[0],uvw
                ,ms,wgt,dirty,epsilon,sigma,sign)

    return dirty

def dirty2vis(uvw, freq, ms, wgt, dirty, fov, epsilon=1e-6,sigma=2):
    """
    Generate Visibility from dirty image by non-uniform fourier transform
    Arguments:
        uvw - 3D coordinates, numpy array, shape - (nrow,3)
        freq - frequencies
        vis - visibility, shape - (nrow,)
        wgt - weight
        nxdirty, nydirty - image size
        fov - field of view
        epsilon - tolerance of relative error (expect, default 1e-6)
        sigma - upsampling factor for grid (default 2)
    Return:
        vis - shape-[M,]
    """
    nrow = uvw.shape[0]
    nxdirty = dirty.shape[0]
    nydirty = dirty.shape[1]
    # u = np.ctypeslib.as_ctypes(uvw[:,0])
    # v = np.ctypeslib.as_ctypes(uvw[:,1])
    # w = np.ctypeslib.as_ctypes(uvw[:,2])
    sign = 1
    if(wgt is None):
        dirty2ms_1(nrow,nxdirty,nydirty,fov,freq[0],uvw
            ,ms,dirty,epsilon,sigma,sign)
    else:
        dirty2ms_2(nrow,nxdirty,nydirty,fov,freq[0],uvw
            ,ms,wgt,dirty,epsilon,sigma,sign)
    return ms
