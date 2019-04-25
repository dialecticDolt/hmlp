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

N = 10000
d = 3

config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.001, 0.01, False)

np.random.seed(10)
class_1 = np.random.randn(d, (int)(np.floor(N/2))) + 2
class_2 = np.random.randn(d, (int)(np.ceil(N/2)))
test_points = np.concatenate((class_1, class_2), axis=1)

GIDS_Owned = getCBLKOwnership(N, rank, nprocs)
CBLK_points = test_points[:, GIDS_Owned]
CBLK_points = np.asarray(CBLK_points, dtype='float32', order='F')
sources = PyGOFMM.PyDistData_CBLK(comm, d, N, darr=CBLK_points)
K = PyGOFMM.KernelMatrix(comm, sources, conf=config)
K.compress()
classes = alg.KMeans(K, 2, maxiter=20, gids=GIDS_Owned)
print(classes)

recvbuf = comm.allgather(classes)
classes = np.concatenate(recvbuf, axis=0).astype('int32')

recvbuf = comm.allgather(GIDS_Owned)
gids = np.concatenate(recvbuf, axis=0).astype('int32')

points = test_points[:, gids].T

plt.scatter(points[:, 0], points[:, 1], c=classes.flatten())
plt.show()

#ones = np.ones([N, d], dtype='float32', order='F')
#test = PyGOFMM.PyDistData_RIDS(comm, N, d, darr=ones, iset=GIDS_Owned)
#alg.testDistDataArray(test)
#print(test.toArray())


rt.finalize()
