import mpi4py
import pygofmm.core as PyGOFMM
import pygofmm.mltools.algorithms as alg

from mpi4py import MPI
import numpy as np

import matplotlib.pyplot as plt

print("At the very top of the script", flush=True)

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

rt = PyGOFMM.PyRuntime()
rt.init_with_MPI(comm)

print("Starting", flush=True)
d = 2
n_local = 200

np.random.seed(10)
points = np.asarray(np.concatenate( (np.random.randn(d, n_local)-5, np.random.randn(d, n_local), np.random.randn(d, n_local)+10), axis=1), dtype='float32')
n_local = (int)(600/nprocs)
points = points[:, rank*n_local:(rank+1)*n_local]

print(rank, np.shape(points), flush=True)

centers = alg.distributed_kmeans_pp(comm, points, 3)

print(np.asarray(centers))


#plt.scatter(points[0, :], points[1, :])
#plt.scatter(centers[0, :], centers[1, :])
#plt.show()
