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


def getCBLKOwnership(N, rank, nprocs):
    cblk_idx = np.asarray(np.arange(rank, N, nprocs), dtype='int32', order='F')
    return cblk_idx

N = 10000
n_per = N/nprocs
d = 5

points = np.fromfile("points_"+str(rank)+".bin", dtype='float32').reshape((d, n_per), order='F')

GIDS_Owned = getCBLKOwnership(N, rank, nprocs)
start_setup = MPI.Wtime()
sources = PyGOFMM.PyDistData_CBLK(comm, m=d, n=N,fileName="points.bin")
config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 256, 64, 128, 0.001, 0.01, False)
K = PyGOFMM.KernelMatrix(comm, sources, conf=config, bandwidth=1)
end_setup = MPI.Wtime()

start_compress = MPI.Wtime()
K.compress()
end_compress = MPI.Wtime()

#Run kernel k-means
nclasses = 2
start_clustering = MPI.Wtime()
classes = PyGOFMM.FastKKMeans(K, nclasses, maxiter=40, gids=GIDS_Owned)
end_clustering = MPI.Wtime()

classes = np.asarray(classes, dtype='int32').flatten()
true_classes = np.asarray(np.fromfile('classes_'+str(rank)+".bin", dtype='float32'), dtype='int32').flatten()
print(true_classes)
print(classes)

print(alg.NMI(comm, true_classes, classes, nclasses))

print("Setup Time: ", end_setup - start_setup)
print("Compress Time: ", end_compress - start_compress)
print("Clustering Time: ", end_clustering - start_clustering)

plt.scatter(points[2, :], points[1, :], c=classes)
plt.show()

rt.finalize()
