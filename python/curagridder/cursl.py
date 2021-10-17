# Contents in this file are specified for RASCIL

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
ms2dirtyf_2.argtypes = [c_int, c_int, c_int, c_float, c_float, np.ctypeslib.ndpointer(np.float32, flags='C'),
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

#----------------------------------------
# the interfaces below are idential to NIFTY
#-----------------------------------------

def ms2dirty(uvw, freq, ms, wgt, nxdirty, nydirty, rad_pix_x, rad_pix_y, nx, ny, epsilon, do_wstacking, *args):
    """
    Generate an image from visibility by non-uniform fourier transform
    Arguments:
        uvw - 3D coordinates, numpy array, shape - (nrow,3)
        freq - frequencies
        ms - visibility, shape - (nrow,)
        wgt - weight
        nxdirty, nydirty - image size
        deg_pix_ - degree per pixel
        epsilon - tolerance of relative error (expect, default 1e-6)
        do_wstacking - True, improved w stacking.
    
    Return:
        dirty image - shape-[nxdirty,nydirty]
    """
    nrow = uvw.shape[0]
    sigma = 2
    fov = rad_pix_x * nxdirty * 180 / np.pi
    sign = -1
    

    if(wgt is None):
        if ms.dtype == np.complex128:
            dirty = np.zeros((nxdirty,nydirty),dtype=np.complex128)
            ms2dirty_1(nrow,nxdirty,nydirty,fov,freq[0],uvw.astype(np.float64)
                ,ms,dirty,epsilon,sigma,sign)
        else:
            dirty = np.zeros((nxdirty,nydirty),dtype=np.complex64)
            ms2dirtyf_1(nrow,nxdirty,nydirty,fov,freq[0],uvw.astype(np.float32)
                ,ms,dirty,epsilon,sigma,sign)
    else:
        if ms.dtype == np.complex128:
            dirty = np.zeros((nxdirty,nydirty),dtype=np.complex128)
            ms2dirty_2(nrow,nxdirty,nydirty,fov,freq[0],uvw.astype(np.float64)
                    ,ms,wgt,dirty,epsilon,sigma,sign)
        else:
            dirty = np.zeros((nxdirty,nydirty),dtype=np.complex64)
            ms2dirtyf_2(nrow,nxdirty,nydirty,fov,freq[0],uvw.astype(np.float32)
                    ,ms,wgt,dirty,epsilon,sigma,sign)
    dirty = np.reshape(dirty,[nxdirty,nydirty])
    return dirty.real

def dirty2ms(uvw, freq, dirty, wgt, rad_pix_x, rad_pix_y, nx, ny, epsilon, do_wstacking, *args):
    """
    Generate Visibility from dirty image by non-uniform fourier transform
    Arguments:
        uvw - 3D coordinates, numpy array, shape - (nrow,3)
        freq - frequencies
        ms - visibility, shape - (nrow,)
        wgt - weight
        nxdirty, nydirty - image size
        fov - field of view
        epsilon - tolerance of relative error (expect, default 1e-6)
        sigma - upsampling factor for grid (default 1.25)
    Return:
        vis - shape-[M,]
    """
    nrow = uvw.shape[0]
    nxdirty = dirty.shape[0]
    nydirty = dirty.shape[1]
    sigma = 2
    fov = rad_pix_x * nxdirty * 180 / np.pi
    sign = -1
    if dirty.dtype==np.float64:
        uvw = uvw.astype(np.float64)
        ms = np.zeros((nrow,1),dtype=np.complex128)
        dirty1 = np.zeros(dirty.shape,dtype=np.complex128)
    else:
        uvw = uvw.astype(np.float32)
        ms = np.zeros((nrow,1),dtype=np.complex64)
        dirty1 = np.zeros(dirty.shape,dtype=np.complex64)
    dirty1.real = dirty

    if(wgt is None):
        if dirty.dtype==np.float64:
            dirty2ms_1(nrow,nxdirty,nydirty,fov,freq[0],uvw
                ,ms,dirty1,epsilon,sigma,sign)
        else:
            dirty2msf_1(nrow,nxdirty,nydirty,fov,freq[0],uvw
                ,ms,dirty1,epsilon,sigma,sign)
    else:
        if dirty.dtype==np.float64:
            dirty2ms_2(nrow,nxdirty,nydirty,fov,freq[0],uvw
                ,ms,wgt,dirty1,epsilon,sigma,sign)
        else:
            dirty2msf_2(nrow,nxdirty,nydirty,fov,freq[0],uvw
                ,ms,wgt,dirty1,epsilon,sigma,sign)
    return ms