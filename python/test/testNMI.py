import pygofmm.core as PyGOFMM
import pygofmm.mltools.algorithms as alg

from mpi4py import MPI
import numpy as np

import matplotlib.pyplot as plt

print("starting script", flush=True)
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

rt = PyGOFMM.Runtime()
rt.init_with_MPI(comm)

N = 300
nclasses = 3
a = np.asarray(np.random.randint(1, nclasses+1, size=N), dtype='int32')
b = np.copy(a)
b[10] = 2
b[20] = 1

ans = alg.NMI(comm, a, b, nclasses)
print(ans)

ans = alg.ChenhanNMI(comm, a, b, nclasses, nclasses)
print(ans)
