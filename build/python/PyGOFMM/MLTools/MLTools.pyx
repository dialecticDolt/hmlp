#Import our functions
from DistData cimport *
from Runtime cimport *
from DistMatrix cimport *
from CustomKernel cimport *
from Config cimport *
from DistTree cimport *
from DistTree cimport Compress as c_compress
from DistInverse cimport *

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
from libc.stdlib cimport malloc, free
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

def testFunction(PyRuntime testparam):
    testparam.init()
    

#KernelKMeans
#Run main loop
def KernelKMeans(MPI.Comm comm, float[::1, :] points, k, float[:] classlist=None):
    a = 2
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
