from pygofmm.core import *
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()



d = 3
N = 10
data = np.random.randn(d, N)
data = np.asarray(data, dtype='float32', order='F')
ids = np.asarray(np.arange(1, N+1), dtype='int32', order='F')
a = PyDistData_RIDS(comm, N, d, iset = ids, darr=data)
b = PyDistData_RIDS(comm, N, d, iset = ids, darr=data)
a.mult(b)
print(a.toArray())

