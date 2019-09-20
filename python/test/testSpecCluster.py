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
d = 2

config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.00001, 0.01, False)

#np.random.seed(10)
#t = np.linspace(0, 2*3.1415, N/2)
#class_1 = [np.sin(t)*5, np.cos(t)*5]
#class_2 = [np.sin(t), np.cos(t)]
#test_points = np.concatenate((class_1, class_2), axis=1)

np.random.seed(10)
class_1 = np.random.randn(d, (int)(np.floor(N/3))) + 10
class_2 = np.random.randn(d, (int)(np.floor(N/3)))
class_3 = np.random.randn(d, (int)(np.ceil(N/3))) - 50
test_points = np.asarray(np.concatenate((class_1, class_2, class_3), axis=1), dtype='float32')

#test_points = np.random.randn(d, N)
GIDS_Owned = getCBLKOwnership(N, rank, nprocs)
print(len(GIDS_Owned))
CBLK_points = test_points[:, GIDS_Owned]
CBLK_points = np.asarray(CBLK_points, dtype='float32', order='F')
sources = PyGOFMM.PyDistData_CBLK(comm, d, N, darr=CBLK_points)
K = PyGOFMM.KernelMatrix(comm, sources, conf=config, bandwidth=1)
K.compress()
classes = alg.SpecCluster(K, 3, gids=GIDS_Owned)
print(np.asarray(classes))

#recvbuf = comm.allgather(classes)
#classes = np.concatenate(recvbuf, axis=0).astype('int32')

#recvbuf = comm.allgather(GIDS_Owned)
#gids = np.concatenate(recvbuf, axis=0).astype('int32')

#points = test_points[:, gids].T
classes = np.asarray(classes, dtype='int32')
plt.scatter(CBLK_points[0, :], CBLK_points[1, :], c=classes.flatten())
plt.show()

#ones = np.ones([N, d], dtype='float32', order='F')
#test = PyGOFMM.PyDistData_RIDS(comm, N, d, darr=ones, iset=GIDS_Owned)
#alg.testDistDataArray(test)
#print(test.toArray())


rt.finalize()
