import pygofmm.core as PyGOFMM
import pygofmm.mltools.algorithms as alg

from mpi4py import MPI
import numpy as np

import matplotlib.pyplot as plt

def getCBLKOwnership(N, rank, nprocs):
    cblk_idx = np.asarray(np.arange(rank, N, nprocs), dtype='int32', order='F')
    return cblk_idx

# Communication init
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

rt = PyGOFMM.PyRuntime()
rt.init_with_MPI(comm)


# data params
N = 10000
d = 3

# GoFMM params
config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.001, 0.01, False)

# Create data(global)
np.random.seed(10)
class_1 = np.random.randn(d, (int)(np.floor(N/2))) + 2
class_2 = np.random.randn(d, (int)(np.ceil(N/2)))
test_points = np.concatenate((class_1, class_2), axis=1)

# create class membership
c1 = np.ones( (int) (np.floot(N/2)) )
c2 = np.ones( (int) (np.floot(N/2)) ) + 1
class_mem_glob = np.concatenate(  (c1, c2), axis = 0)

# extract owned data and set up cblk object
GIDS_Owned = getCBLKOwnership(N, rank, nprocs)
CBLK_points = test_points[:, GIDS_Owned]
CBLK_points = np.asarray(CBLK_points, dtype='float32', order='F')
sources = PyGOFMM.PyDistData_CBLK(comm, d, N, darr=CBLK_points)
class_mem = class_mem_glob[GIDS_Owned]

# Create + compress kernel
K = PyGOFMM.KernelMatrix(comm, sources, conf=config)
K.compress()

# Call KDE
density = alg.KDE(K, 2,GIDS_Owned,class_mem)
print(density)


# plot!
#recvbuf = comm.allgather(density)
#density = np.concatenate(recvbuf, axis=0).astype('int32')
#
#recvbuf = comm.allgather(GIDS_Owned)
#gids = np.concatenate(recvbuf, axis=0).astype('int32')
#
#points = test_points[:, gids].T
#
#plt.scatter(points[:, 0], points[:, 1], c=classes.flatten())
#plt.show()


rt.finalize()




