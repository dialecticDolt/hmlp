#distutils: language = c++
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
cimport  mpi4py.libmpi as libmpi

from DistData cimport *
from DistTree cimport *


cdef extern from "/workspace/will/dev/hmlp/gofmm/igofmm_mpi.hpp" namespace "hmlp::mpigofmm" nogil:

    cdef void DistFactorize[T, TREE](TREE&, T) except +
    cdef void DistSolve[T, TREE](TREE&,Data[T]&) except +
    cdef void ComputeError[TREE, T](TREE&, T, Data[T], Data[T])
     



