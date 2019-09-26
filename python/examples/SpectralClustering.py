import pygofmm.core as PyGOFMM
import pygofmm.mltools as MLTools
import pygofmm.mltools.algorithms as FMML

from mpi4py import MPI
import numpy as np
import argparse
import sys


comm = MPI.Comm.Clone(MPI.COMM_WORLD)
nprocs = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Cluster with Kernel K-Means')

parser.add_argument('-file', type=str, required=False, default=None,
        help="Specify the file to load the data")

parser.add_argument('-truth', type=str, required=False, default=None, help="Specify the file to load truth values")

parser.add_argument('-N', type=int, required=False, default= 10000, help="Specify the size of the random dataset. Set --scaling strong or weak")

parser.add_argument('--scaling', type=str, dest='scaling', choices=['strong', 'weak'],default='weak', help="Scaling of the point set size. weak = N*p, strong=N")

parser.add_argument('-nclasses', type=int, required=False, default=2, help="Specify the number of classes to generate")

parser.add_argument('-d', type=int, required=False, default = 5, help="Specify the dimension of the random point set")

parser.add_argument('-leaf', type=int, required=False, default = 128, help="Specify the leaf size of the kd-tree")

parser.add_argument('-k', type=int, required=False, default = 64, help="Specify the number of nearest neighbors to use in determining the far field and sparse corrections")

parser.add_argument('-max_rank', type=int, required=False, default=128, help="Specify the maximum rank of the internal nodes")

parser.add_argument("-tolerance", type=float, required=False, default = 1e-3, help="Specify the accuracy tolarance of the internal low-rank approximations")

parser.add_argument("-budget", type=float, required=False, default = 0, help="Specify the budget of the exact evaulation close neighbor list")

parser.add_argument("-iter", type=int, dest="maxiter", required=False, default=20, help="Specify the maximum number of iterations for kmeans")

parser.add_argument("-kmeans_init", type=str, dest="init", required=False, default="random", choices=['random', '++'], help="Specify the initialization of k-means. Used in Kernel K-Means or in the post processing of Spectral Clustering")

parser.add_argument("-secure", type=str2bool, required=False, default=True, help="Set Secure Accuracy (i.e. use level restriction)")

parser.add_argument("-bandwidth", type=float, required=False, default = 1, help="Specify the bandwidth for the similarity function")

args = parser.parse_args()

leaf_node_size = args.leaf
neighbors = args.k
maximum_rank = args.max_rank
tol = args.tolerance
budget = args.budget

rt = PyGOFMM.Runtime()
rt.initialize(comm)

if args.file:
    #Option 1: Load a data set
    N = 10
else:
    #Option 2: Construct a random point set

    if args.scaling=='strong':
        N = args.N
    elif args.scaling=='weak':
        N = args.N*nprocs

    d = args.d

    #Construct the point set
    #np.random.seed(10)

    extra = (N % args.nclasses)
    for i in range(args.nclasses):
        offset = np.random.rand(d, 1)*10
        print(offset)
        if(extra):
            split = 1
            extra = extra - 1
        else:
            split = 0
        new_class = np.random.randn(d, (int)(np.floor(N/args.nclasses)) + split) + offset
        new_truth = np.ones(np.shape(new_class)) + i
        if i == 0:
            point_set = new_class
            truth_set = new_truth
        else:
            point_set = np.concatenate((point_set, new_class), axis=1)
            truth_set = np.concatenate((truth_set, new_truth), axis=1)

    point_set = PyGOFMM.reformat(point_set)

    #Redistribute points to cyclic partition
    start_s = MPI.Wtime()
    sources, gids_owned = PyGOFMM.distribute_cblk(comm, point_set)


#Setup and compress the kernel matrix
setup_time = MPI.Wtime()
config = PyGOFMM.Config("GEOMETRY_DISTANCE", N, leaf_node_size, neighbors, maximum_rank, tol, budget, args.secure)
K = PyGOFMM.KernelMatrix(comm, sources, config=config, bandwidth=args.bandwidth)
setup_time = MPI.Wtime() - setup_time

compress_time = MPI.Wtime()
K.compress()
compress_time = MPI.Wtime() - compress_time

#Run spectral clustering
print(N)
clustering_time = MPI.Wtime()
clustering_output = FMML.SpectralCluster(K, args.nclasses, gids_owned)
clustering_time = MPI.Wtime() - clustering_time

local_class_assignments = clustering_output.classes

#classes, gids = FMML.gather_points(test)
#points = test_points[:, gids].T

truth_set = np.asarray(truth_set[0, gids_owned], dtype='int32').flatten()
local_class_assignments = np.asarray(local_class_assignments, dtype='int32').flatten()

print(FMML.NMI(comm, truth_set, local_class_assignments, args.nclasses))
print(FMML.ChenhanNMI(comm, truth_set, local_class_assignments, args.nclasses, args.nclasses))

rt.finalize()
