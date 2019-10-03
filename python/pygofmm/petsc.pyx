# cython: boundscheck=False

#Import from PyGOFMM
cimport pygofmm.core as PyGOFMM
import pygofmm.core as PyGOFMM

from pygofmm.DistData cimport *
from pygofmm.CustomKernel cimport *

#Import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
from mpi4py import MPI

#Import from cython: cpp
from libcpp.pair cimport pair
from libcpp cimport vector
from libcpp.map cimport map

#Import from cython: c
from libc.math cimport sqrt, log2
from libc.string cimport strcmp
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.stdio cimport printf
from libcpp cimport bool as bool_t

#Import from cython
import cython
from cython.operator cimport dereference as deref
from cython.view cimport array as cvarray
from cython.parallel cimport prange

#Numpy
import numpy as np
cimport numpy as np
np.import_array()

#PETSc and SLEPc
import petsc4py
from slepc4py import SLEPc
from petsc4py import PETSc

#Imports for flush
import sys
import time


cdef class GOFMM_Handler(object):
    cdef float reg

    def __init__(self, comm, N, d, sources, targets=None, kstring="GAUSSIAN", config=None, petsc=True, bwidth=1.0, regularization = 1.0):
        self.rt = PyGOFMM.Runtime()
        self.rt.init_with_MPI(comm)
        self.reg = regularization
        self.comm = comm
        if not petsc:
            self.K = PyGOFMM.KernelMatrix(comm, sources, targets, kstring, config, bwidth)
            self.size = N
        if petsc:
            if(targets==None):
                with sources as src:
                    GOFMM_src = PyGOFMM.DistData_CBLK(comm, d, N, arr=src.astype('float32'))
                    self.K = PyGOFMM.Kernelmatrix(comm, GOFMM_src, targets, kstring, config, bwidth)
            else:
                with sources as src:
                    with targets as trg:
                        GOFMM_src = PyGOFMM.DistData_CBLK(comm, d, N, arr=src.astype('float32'))
                        GOFMM_trg = PyGOFMM.DistData_CBLK(comm, d, N, arr=trg.astype('float32'))
                        self.K = PyGOFMM.KernelMatrix(comm, GOFMM_src, GOFMM_trg, kstring, config, bwidth)
        self.nlocal = len(self.K.get_rids())
        self.K.compress()

    def __dealloc__(self):
        self.rt.finalize()

    def getSize(self):
        return self.size

    def getLocal(self):
        return self.nlocal

    def getMPIComm(self):
        return self.comm

    def mult(self, mat, X, Y):
        cdef int i, length
        cdef float[:, :] res
        cdef float[:] c_x
        cdef double[:] c_y 

        with X as x:
            with Y as y:
                n = len(x)
                nrhs = 1
                c_x = PyGOFMM.reformat(x)
                GOFMM_x = PyGOFMM.DistData_RIDS(self.comm, m=self.size, n=nrhs, tree=self.K.get_tree(), arr=c_x)
                GOFMM_y = self.K.evaluate(GOFMM_x)
                res = GOFMM_y.to_array()
                length = len(y)
                c_y = y
                for i in prange(length, nogil=True):
                    c_y[i] = res[i, 0]
        return Y

    def mult_hmlp(self, GOFMM_x):
        GOFMM_y = self.K.evaluate(GOFMM_x)
        return GOFMM_y

    def solve_hmlp(self, GOFMM_b, l):
        GOFMM_x = GOFMM_b #TODO: Placeholder, replace this with a deep copy
        self.K.solve(GOFMM_b, l)

    def get_gids(self):
        gids = self.K.get_tree().get_gids()
        return gids

    def redistribute_hmlp(self, GOFMM_x):
        n = GOFMM_x.rows()
        d = GOFMM_x.cols()
        GOFMM_y = PyGOFMM.DistData_RIDS(self.comm, n, d, tree=self.K.get_tree())
        GOFMM_y.redistribute(GOFMM_x)
        return GOFMM_y

    def redistribute_any_hmlp(self, GOFMM_x, rids):
        n = GOFMM_x.rows()
        d = GOFMM_x.cols()
        GOFMM_y = PyGOFMM.DistData_RIDS(self.comm, n, d, iset=rids)
        GOFMM_y.redistribute(GOFMM_x)
        return GOFMM_y

    def redistribute(self, X, d, n, nper, form="petsc", transpose=True):
        with X as x:
            if(transpose):
                x = x.reshape(nper, d, order='C')
            x = np.asfortranarray(x)
            GOFMM_x = PyGOFMM.DistData_RIDS(self.comm, n, d, darr=x.astype('float32'))
            GOFMM_y = self.redistribute_hmlp(GOFMM_x)
            if(form=="petsc"):
                return PETSc.Vec().createWithArray(np.copy(GOFMM_y.to_array()))
            elif(form=="hmlp"):
                return GOFMM_y
            else:
                raise Exception("Redistribute must return either a PETSc Vec() or HMLP DistData<RIDS, STAR, T>")

    def redistribute_any(self, X, d, n, nper, target_rids, form="petsc", transpose=True):
        with X as x:
            if (transpose):
                x = x.reshape(nper, d, order='C')
            x = np.asfortranarray(x)
            GOFMM_x = PyGOFMM.DistData_RIDS(self.comm, n, d, darr=x.astype('float32'))
            GOFMM_y = self.redistribute_any_hmlp(GOFMM_x, target_rids)
            if(form=="petsc"):
                return PETSc.Vec().createWithArray(np.copy(GOFMM_y.to_array()))
            elif(form=="hmlp"):
                return GOFMM_y
            else:
                raise Exception("Redistribute must returen either a PETSc Vec() or HMLP DistData<RIDS, STAR, T>")



    def getDiagonal(self, mat, result):
        local_gids = self.K.get_tree().get_gids()
        y = np.empty([self.size, 1], dtype='float32')
        with result as y:
            #TODO: Parallelize this
            for i in range(len(local_gids)):
                y[i] = self.K.get_value(local_gids[i], local_gids[i])
        return result

    def solve(self, mat, B, X):
        cdef int n, nrhs, i
        cdef int length
        cdef double[:] c_x
        cdef float[:, :] res
        with X as x:
            with B as b:
                n = len(b)
                nrhs = 1
                GOFMM_b = PyGOFMM.LocalData(m=n, n=nrhs, arr=x.astype('float32'))
                self.K.solve(GOFMM_b, self.reg)
                res = GOFMM_b.to_array()
                length = len(x)
                c_x = x
                for i in prange(length, nogil=True):
                    c_x[i] = res[i, 0]

cdef class Kernel_Handler(object):

    cdef PyGOFMM.KernelMatrix K
    cdef MPI.Comm comm
    cdef int size
    cdef PyGOFMM.DistData_RIDS D
    cdef int[:] rids
    cdef bool_t normalize
    cdef float reg
    cdef int n_local

    def __init__(self, K, normalize=False, regularization=1.0):
        self.K = K
        self.reg = regularization
        self.size = K.get_size()
        self.comm = K.get_comm()
        self.rids = K.get_tree().get_gids().astype('int32')
        self.n_local = len(self.rids)
        self.normalize = normalize
        if self.normalize:
            local_ones = np.ones(self.n_local, dtype='float32', order='F')
            ones = PyGOFMM.DistData_RIDS(self.comm, self.size, 1, arr=local_ones, iset=self.rids)
            self.D = self.K.evaluate(ones)
    
    def getSize(self):
        return self.size

    def getMPIComm(self):
        return self.comm

    def getLocal(self):
        return self.n_local

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def mult(self, mat, X, Y):
        cdef float[:, :] npD
        cdef float[:, :] res
        cdef double[:] c_x
        cdef double[:] c_y
        cdef float[:] normedx 
        cdef int n, nrhs
        cdef int length, i
        cdef bool_t normalize
        normalize = self.normalize
        
        if normalize:
            npD = self.D.to_array()
            with X as x:
                with Y as y:
                    n = len(x)
                    nrhs = 1

                    #TODO: Make sure this doesn't copy
                    c_x = x
                    c_y = y

                    normedx = np.zeros(np.shape(x), order='F', dtype='float32')
                    
                    length = len(x)
                    for i in prange(length, nogil=True):
                        normedx[i] = c_x[i]
                        normedx[i] = 1/sqrt(npD[i, 0]) * c_x[i]

                    #for i in range(length):
                    #    normedx[i] = 1/sqrt(npD[i, 0]) * c_x[i]
                    
                    GOFMM_x = PyGOFMM.DistData_RIDS(self.comm, m=self.size, n=nrhs, tree=self.K.get_tree(), arr=normedx)
                    GOFMM_y = self.K.evaluate(GOFMM_x)
                    res = GOFMM_y.to_array()
                    
                    length = len(y)
                    for i in prange(length, nogil=True):
                        c_y[i] = 1/sqrt(npD[i, 0]) * res[i, 0]
                        #c_y[i] = 1/npD[i, 0] * res[i, 0]

            #        for i in range(length):
            #                y[i] = 1/sqrt(npD[i, 0]) * res[i, 0]
        else:
            with X as x:
                with Y as y:

                    #c_y = y
                    
                    n = len(x)
                    nrhs = 1
                    GOFMM_x = PyGOFMM.DistData_RIDS(self.comm, m=self.size, n=nrhs, tree=self.K.get_tree(), arr=x.astype('float32'))
                    GOFMM_y = self.K.evaluate(GOFMM_x)
                    res = GOFMM_y.to_array()
                    #print(np.array(GOFMM_y.to_array()))

                    #TODO: Parallelize this
                    length = len(y)
                    #for i in prange(length, nogil=True):
                    for i in range(length):
                        y[i] = res[i, 0]

        return Y

    def mult_hmlp(self, GOFMM_x):
        GOFMM_y = self.K.evaluate(GOFMM_x)
        return GOFMM_y

    def solve_hmlp(self, GOFMM_b, l):
        GOFMM_x = GOFMM_b #TODO: Placeholder, replace this with a deep copy
        self.K.solve(GOFMM_b, l)

    def get_gids(self):
        gids = self.K.get_tree().get_gids()
        return gids

    def redistribute_hmlp(self, GOFMM_x):
        n = GOFMM_x.rows()
        d = GOFMM_x.cols()
        GOFMM_y = PyGOFMM.DistData_RIDS(self.comm, n, d, tree=self.K.get_tree())
        GOFMM_y.redistribute(GOFMM_x)
        return GOFMM_y

    def redistribute_any_hmlp(self, GOFMM_x, rids):
        n = GOFMM_x.rows()
        d = GOFMM_x.cols()
        GOFMM_y = PyGOFMM.DistData_RIDS(self.comm, n, d, iset=rids)
        GOFMM_y.redistribute(GOFMM_x)
        return GOFMM_y

    def redistribute(self, X, d, n, nper, form="petsc", transpose=True):
        with X as x:
            if(transpose):
                x = x.reshape(nper, d, order='C')
            x = np.asfortranarray(x)
            GOFMM_x = PyGOFMM.DistData_RIDS(self.comm, n, d, darr=x.astype('float32'))
            GOFMM_y = self.redistribute_hmlp(GOFMM_x)
            if(form=="petsc"):
                return PETSc.Vec().createWithArray(np.copy(GOFMM_y.to_array()))
            elif(form=="hmlp"):
                return GOFMM_y
            else:
                raise Exception("Redistribute must return either a PETSc Vec() or HMLP DistData<RIDS, STAR, T>")

    def redistribute_any(self, X, d, n, nper, target_rids, form="petsc", transpose=True):
        with X as x:
            if (transpose):
                x = x.reshape(nper, d, order='C')
            x = np.asfortranarray(x)
            GOFMM_x = PyGOFMM.DistData_RIDS(self.comm, n, d, darr=x.astype('float32'))
            GOFMM_y = self.redistribute_any_hmlp(GOFMM_x, target_rids)
            if(form=="petsc"):
                return PETSc.Vec().createWithArray(np.copy(GOFMM_y.to_array()))
            elif(form=="hmlp"):
                return GOFMM_y
            else:
                raise Exception("Redistribute must return either a PETSc Vec() or HMLP DistData<RIDS, STAR, T>")

    #NOTE: Must be camel case to match what petsc4py expects
    def getDiagonal(self, mat, result):
        local_gids = self.K.get_tree().get_gids()
        y = np.empty([self.size, 1], dtype='float32')
        with result as y:
            #TODO: Parallelize this
            for i in range(len(local_gids)):
                y[i] = self.K.get_value(local_gids[i], local_gids[i])
        return result

    def solve(self, mat, B, X):
        cdef int n, nrhs, length, i
        cdef double[:] c_x
        cdef float[:, :] res
        with X as x:
            with B as b:
                n = len(b)
                nrhs = 1
                GOFMM_b = PyGOFMM.LocalData(m=n, n=nrhs, arr=x.astype('float32'))
                self.K.solve(GOFMM_b, self.reg)
                res = GOFMM_b.to_array()
                c_x = x
                length = len(x)
                for i in prange(length, nogil=True):
                    c_x[i] = res[i, 0]

def PETSC_Handler(handler, petsc_comm):
    cdef int N, n_local
    N = handler.getSize();
    n_local = handler.getLocal();
    A = PETSc.Mat().createPython(((n_local, N), (n_local, N)), comm = petsc_comm)
    A.setPythonContext(handler)
    A.setUp()
    return A


