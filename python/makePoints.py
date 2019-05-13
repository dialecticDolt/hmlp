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

N = 10000
d = 5

#np.random.seed(10)
#t = np.linspace(0, 2*3.1415, N/2)
#class_1 = [np.sin(t)*5, np.cos(t)*5]
#class_2 = [np.sin(t), np.cos(t)]
#test_points = np.concatenate((class_1, class_2), axis=1)

np.random.seed(10)
class_1 = np.random.randn(d, (int)(np.floor(N/3))) + 10
class_2 = np.random.randn(d, (int)(np.floor(N/3)))
class_3 = np.random.randn(d, (int)(np.ceil(N/3)))
test_points = np.asarray(np.concatenate((class_1, class_2, class_3), axis=1), dtype='float32', order='F')

if rank==0:
    test_points.tofile("points.bin")

c1 = np.ones(np.shape(class_1))
c2 = np.ones(np.shape(class_2))+1
c3 = np.ones(np.shape(class_3))+1

true_classes = np.asarray(np.concatenate((c1, c2, c3), axis=1), dtype='float32', order='F')

GIDS_Owned = getCBLKOwnership(N, rank, nprocs)
CBLK_classes = true_classes[0, :]

CBLK_classes = np.asarray(CBLK_classes, dtype='float32', order='F')

CBLK_classes.tofile("classes_"+str(rank)+".bin")

test_points = test_points[:, GIDS_Owned]
test_points.tofile("points_"+str(rank)+".bin")

