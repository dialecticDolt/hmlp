#Import from PyGOFMM
cimport pygofmm.core as PyGOFMM
import pygofmm.core as PyGOFMM #I don't know what this is doing

from pygofmm.DistData cimport *
from pygofmm.CustomKernel cimport *

#Import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

#Import from cython: cpp
from libcpp.pair cimport pair
from libcpp cimport vector
from libcpp.map cimport map

#Import from cython: c
from libc.math cimport sqrt
from libc.string cimport strcmp
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.stdio cimport printf

#Import from cython
import cython
from cython.operator cimport dereference as deref
from cython.view cimport array as cvarray
from cython.parallel cimport prange

#Numpy
import numpy as np
cimport numpy as np
np.import_array()

#REWRITE KKMEANS

def testFunction(PyGOFMM.PyRuntime testparam):
    testparam.init()
    testparam.finalize()


#KernelKMeans
#Assume data already configured in PyGOFMM kernel
#Version: Without BLAS
def KMeans(PyGOFMM.KernelMatrix K,int nclasses, int[:] classvec=None, maxiter=10):
    cdef int N, n_local, c, i, j, k
    N = K.getSize()
    cdef MPI.Comm comm
    comm = K.getComm()
    cdef float[:, :] cI = np.zeros([N, nclasses], dtype='float32', order='F')
    cdef int[:] rids = K.getTree().getGIDS() #TODO: Add GIDS function to K
    n_local = len(rids)
    if classvec is None:
        #initialize class indicator block
        for i in prange(N, nogil=True):
            #Assign random class to each element
            c = 1 + <int>(rand()/RAND_MAX*nclasses)
            cI[i, c] = 1.0
    else:
        for i in prange(N, nogil=True):
            c = classvec[i]
            cI[i, c] = 1.0
    
    #copy class indicator block to DistData object
    cdef float[:, :] local_ones = np.ones([n_local, 1], dtype='float32', order='F')
    Ones = PyGOFMM.PyDistData_RIDS(comm, N, 1, iset=rids, darr=local_ones)
    D = K.evaluate(Ones)

    H = PyGOFMM.PyDistData_RIDS(comm, N, nclasses, iset=rids, darr=cI)
    #Main Loop
    cdef float[:, :] HKH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] HKH = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] HDH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] HDH = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] DKH = np.zeros([n_local, nclasses], dtype='float32', order='F')
    
    cdef float* H_Data
    cdef float* D_Data
    cdef float* KH_Data
    print(type(D.c_data))
    #D_Data = deref(D.c_data).columndata(0)
    #for i in xrange(maxiter):
    #    KH = K.evaluate(H)
    #    H_Data = deref(H.c_data).columndata(0)
    #    KH_Data = KH.c_data.columndata(0)
    #
    #    #TODO: Replace this with MKL calls
    #    for i in xrange(nclasses):
    #        for j in xrange(nclasses):
    #            for r in prange(n_local, nogil=True):
    #                HKH_local[i, j] += H_Data(r, j) * KH_Data(r, i)
    #                HDH_local[i, j] += H_Data(r, j) * KH_Data(r, i)
    #                DKH[i, j] = 1/D_Data(r, 0) * KH_Data(r, j)
    #
    #    
    #    comm.Allreduce(HKH_local, HKH, op=MPI.SUM)
    #    comm.Allreduce(HDH_local, HDH, op=MPI.SUM)



#Alternative appraoch
    #initialize GOFMM
    
    #initialize class indicator vector block


    #start c++ KKMeans script (KKmeansHelper)

#Generate Class inidicator

#KKMeansHelper
#Generate lookup matricies
#compute similarity
#update class vector
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef KKMeansHelper(RIDS_STAR_DistData[float] H, ):
    a = 10
