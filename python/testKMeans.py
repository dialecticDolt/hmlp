import pygofmm.core as PyGOFMM
import pygofmm.mltools.algorithms as alg

from mpi4py import MPI
import numpy as np

import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

rt = PyGOFMM.PyRuntime()
rt.init_with_MPI(comm)

N = 10000
d = 5


#Construct the point set
np.random.seed(10)
class_1 = np.random.randn(d, (int)(np.floor(N/3))) + 10
class_2 = np.random.randn(d, (int)(np.floor(N/3)))
class_3 = np.random.randn(d, (int)(np.ceil(N/3))) - 2
test_points = np.asarray(np.concatenate((class_1, class_2, class_3), axis=1), dtype='float32')

#t = np.linspace(0, 2*3.1415, N/2)
#class_1 = [np.sin(t)*5, np.cos(t)*5]
#class_2 = [np.sin(t), np.cos(t)]
#test_points = np.concatenate((class_1, class_2), axis=1)
#test_points = np.asarray(test_points, dtype='float32')

#Redistribute points to cyclic partition
sources, GIDS_Owned = PyGOFMM.CBLK_Distribute(comm, test_points)
print(len(GIDS_Owned))

#Setup and compress the kernel matrix
config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.00001, 0.03, False)
K = PyGOFMM.KernelMatrix(comm, sources, conf=config, bandwidth=1)
K.compress()

#Run kernel k-means
classes = alg.KKMeans(K, test_points, 3, maxiter=40, gids=GIDS_Owned)

#Redistribute points to original partition


#Gather points to root (for plotting)
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
