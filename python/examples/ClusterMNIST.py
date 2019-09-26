import pygofmm.core as PyGOFMM
import pygofmm.mltools as MLTools
import pygofmm.mltools.algorithms as FMML

from mpi4py import MPI
import numpy as np
import argparse
import sys

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

comm = MPI.Comm.Clone(MPI.COMM_WORLD)
nprocs = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Cluster MNIST')

parser.add_argument('-nclasses', type=int, required=False, default=2, help="Specify the number of classes to generate")

parser.add_argument('-leaf', type=int, required=False, default = 128, help="Specify the leaf size of the kd-tree")

parser.add_argument('-k', type=int, required=False, default = 64, help="Specify the number of nearest neighbors to use in determining the far field and sparse corrections")

parser.add_argument('-max_rank', type=int, required=False, default=128, help="Specify the maximum rank of the internal nodes")

parser.add_argument("-tolerance", type=float, required=False, default = 1e-3, help="Specify the accuracy tolarance of the internal low-rank approximations")

parser.add_argument("-budget", type=float, required=False, default = 0, help="Specify the budget of the exact evaulation close neighbor list")

parser.add_argument("-bandwidth", type=float, required=False, default = 1, help="Specify the bandwidth for the similarity function")

parser.add_argument("-cluster", type=str, required=False, default="sc", help="Specify which clustering method to use")

parser.add_argument("-iter", type=int, dest="maxiter", required=False, default=20, help="Specify the maximum number of iterations for kmeans")

parser.add_argument("-kmeans_init", type=str, dest="init", required=False, default="random", choices=['random', '++'], help="Specify the initialization of k-means. Used in Kernel K-Means or in the post processing of Spectral Clustering")

parser.add_argument("-secure", type=str2bool, required=False, default=True, help="Set Secure Accuracy (i.e. use level restriction)")

args = parser.parse_args()

leaf_node_size = args.leaf
neighbors = args.k
maximum_rank = args.max_rank
tol = args.tolerance
budget = args.budget

rt = PyGOFMM.Runtime()
rt.initialize(comm)

point_set, truth_set = loadlocal_mnist(images_path='/workspace/will/dev/datasets/MNIST60k/train-images-idx3-ubyte', labels_path='/workspace/will/dev/datasets/MNIST60k/train-labels-idx1-ubyte')

point_set = point_set.T
d = point_set.shape[0]
N = point_set.shape[1]

truth_set = truth_set + 1
point_set = PyGOFMM.reformat(point_set)
truth_set = PyGOFMM.reformat(truth_set)

#Redistribute points to cyclic partition
start_s = MPI.Wtime()
sources, gids_owned = PyGOFMM.distribute_cblk(comm, point_set)

starting_assignment = starting_assignment[gids_owned]

#Setup and compress the kernel matrix
setup_time = MPI.Wtime()
config = PyGOFMM.Config("GEOMETRY_DISTANCE", N, leaf_node_size, neighbors, maximum_rank, tol, budget, args.secure)
K = PyGOFMM.KernelMatrix(comm, sources, config=config, bandwidth=args.bandwidth)
setup_time = MPI.Wtime() - setup_time

compress_time = MPI.Wtime()
K.compress()
compress_time = MPI.Wtime() - compress_time

#Run kernel k-means
clustering_time = MPI.Wtime()
if args.cluster == 'sc':
    clustering_output = FMML.SpectralCluster(K, args.nclasses, gids=gids_owned, init=args.init)
else:
    clustering_output = FMML.KernelKMeans(K,  args.nclasses, gids=gids_owned, init=args.init, maxiter=args.maxiter)

clustering_time = MPI.Wtime() - clustering_time

local_class_assignments = clustering_output.classes

truth_set = np.asarray(truth_set[gids_owned], dtype='int32').flatten()
local_class_assignments = np.asarray(local_class_assignments, dtype='int32').flatten()

print(FMML.ChenhanNMI(comm, truth_set, local_class_assignments, 10, args.nclasses))

rt.finalize()
