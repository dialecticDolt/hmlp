import pygofmm.core as PyGOFMM
import pygofmm.mltools.algorithms as alg

from mpi4py import MPI
import numpy as np

import matplotlib.pyplot as plt

print("starting script", flush=True)
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

rt = PyGOFMM.PyRuntime()
rt.init_with_MPI(comm)

N = 30000000
d = 2

config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.001, 0.01, False)

#np.random.seed(10)
class_1 = np.random.randn(d, (int)(np.floor(N/3))) + 10
class_2 = np.random.randn(d, (int)(np.ceil(N/3)))
class_3 = np.random.randn(d, (int)(np.ceil(N/3))) - 5


tc_1 = np.ones([(int)(np.floor(N/3))], dtype='int32')
tc_2 = np.ones([(int)(np.ceil(N/3))], dtype='int32')+1
tc_3 = np.ones([(int)(np.ceil(N/3))], dtype='int32')+2

test_points = np.concatenate((class_1, class_2, class_3), axis=1)
test_points = np.asarray(test_points, dtype='float32', order='F')

true_classes = np.concatenate((tc_1, tc_2, tc_3), axis=0)
true_classes = np.asarray(true_classes, dtype='int32', order='F')

comm.Barrier()
print("Starting kmeans", flush=True)
classes = alg.KMeans(comm, test_points, 3, maxiter=100, init="++")

print(np.asarray(classes), flush=True)
plt.scatter(test_points[0, :], test_points[1, :], c=classes.flatten())
plt.show()
rt.finalize()
