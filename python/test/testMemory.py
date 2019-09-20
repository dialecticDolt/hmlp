import pygofmm.core as PyGOFMM
import pygofmm.mltools.algorithms as alg

from mpi4py import MPI
import numpy as np

import matplotlib.pyplot as plt

def getCBLKOwnership(N, rank, nprocs):
    cblk_idx = np.asarray(np.arange(rank, N, nprocs), dtype='int32', order='F')
    return cblk_idx

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

rt = PyGOFMM.PyRuntime()
rt.init_with_MPI(comm)

N = 100000
d = 3

config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 512, 32, 512, 0.001, 0, True)

np.random.seed(10)

t = np.linspace(0, 2*3.1415, N)
class_1 = [np.sin(t), np.cos(t), t]
test_points = np.asarray(class_1, dtype='float32')

GIDS_Owned = getCBLKOwnership(N, rank, nprocs)
CBLK_points = test_points[:, GIDS_Owned]
CBLK_points = np.asarray(CBLK_points, dtype='float32', order='F')
sources = PyGOFMM.PyDistData_CBLK(comm, d, N, darr=CBLK_points)
K = PyGOFMM.KernelMatrix(comm, sources, conf=config, bandwidth=1)
K.compress()

dmap = alg.MemoryTest(K, 1000)

rt.finalize()
