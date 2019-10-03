import pygofmm.core as PyGOFMM
import pygofmm.mltools.algorithms as alg

from mpi4py import MPI
import numpy as np
import argparse
import sys

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

parser = argparse.ArgumentParser(description='Time simple well seperated gaussians')

parser.add_argument('-N', type=int, required=False, default= 10000, help="Specify the size of the random dataset. Set --scaling strong or weak")

parser.add_argument('--scaling', type=str, dest='scaling', choices=['strong', 'weak'],default='weak', help="Scaling of the point set size. weak = N*p, strong=N")

parser.add_argument('-d', type=int, required=False, default = 5, help="Specify the dimension of the random point set")

parser.add_argument('-leaf', type=int, required=False, default = 128, help="Specify the leaf size of the kd-tree")

parser.add_argument('-k', type=int, required=False, default = 64, help="Specify the number of nearest neighbors to use in determining the far field and sparse corrections")

parser.add_argument('-max_rank', type=int, required=False, default=128, help="Specify the maximum rank of the internal nodes")

parser.add_argument("-tolerance", type=float, required=False, default = 1e-5, help="Specify the accuracy tolarance of the internal low-rank approximations")

parser.add_argument("-budget", type=float, required=False, default = 0, help="Specify the budget of the exact evaulation close neighbor list")

parser.add_argument("-iter", type=int, dest="iter", required=False, default=20, help="Specify the maximum number of iterations for kmeans")

parser.add_argument("-init", type=str, dest="init", required=False, default="random", choices=['random', '++'], help="Specify the initialization of k-means. Used in Kernel K-Means or in the post processing of Spectral Clustering")

parser.add_argument("-secure", type=str2bool, required=False, default=False, help="Set Secure Accuracy (i.e. use level restriction)")

parser.add_argument("-slack", type=int, dest="slack", required=False, default=0, help="Specify how many extra (total: nclasses+slack) eigenvectors to keep for spectral clustering")

args = parser.parse_args()

leaf_node_size = args.leaf
neighbors = args.k
maximum_rank = args.max_rank
tol = args.tolerance
budget = args.budget

rt = PyGOFMM.Runtime()
rt.init_with_MPI(comm)

N = args.N
if args.scaling == 'weak':
    N = N*nprocs;
d = args.d
nclasses = 2

#Construct the point set
np.random.seed(10)
extra = (N%nclasses)
offset = np.array([5, 0])
for i in range(nclasses):
    if(extra):
        split = 1
        extra = extra - 1
    else:
        split = 0
    new_class = np.random.randn(d, (int)(np.floor(N/nclasses))+split) + offset[i]
    new_truth = np.ones(new_class.shape[1]) + i
    if i == 0:
        point_set = new_class
        truth_set = new_truth
    else:
        point_set = np.concatenate((point_set, new_class), axis=1)
        truth_set = np.concatenate((truth_set, new_truth), axis=0)

point_set = PyGOFMM.reformat(point_set)

#Redistribute points to cyclic partition
sources, GIDS_Owned = PyGOFMM.distribute_cblk(comm, point_set)

true_classes = np.asarray(truth_set[GIDS_Owned], dtype='int32').flatten()

a = 0.2
b = 5
bandwidths = np.logspace(np.log2(a), np.log2(b), num=4, base=2)
bandwidths = np.array([5])
spec = True;
if spec:
        t = "spectral"
else:
        t = "kernel"

L = args.leaf #leafnodesize
k = args.k #number of neighbors
tol = args.tolerance
budget = args.budget
sec = args.secure


results = open(t+"_clustering_tol"+str(tol)+"_budget"+str(budget)+"_MPI_"+str(nprocs)+".txt", "a")


for lam in bandwidths:
        #Setup and compress the kernel matrix
        start_s = MPI.Wtime()
        config = PyGOFMM.Config("GEOMETRY_DISTANCE", N, L, k, L, tol, budget, sec)
        K = PyGOFMM.KernelMatrix(comm, sources, config=config, bandwidth=lam)

        end_s = MPI.Wtime()
        start_co = MPI.Wtime()
        K.compress()
        end_co = MPI.Wtime()

        #Run Clustering
        start_c = MPI.Wtime()
        if spec:
            output  = alg.SpectralCluster(K, nclasses, gids=GIDS_Owned, init=args.init, slack=args.slack, maxiter=args.iter)
        else:
            output = alg.KernelKMeans(K,  nclasses, maxiter=args.iter, init=args.init, gids=GIDS_Owned)
        end_c = MPI.Wtime()
        classes = output.classes
        classes = np.asarray(classes, dtype='int32').flatten()

        print("Bandwidth", lam)
        print("NMI:", alg.ChenhanNMI(comm, true_classes, classes, nclasses, nclasses))
        print("ARI:", alg.ARI(comm, true_classes, classes, nclasses, nclasses))
        print("Setup: ", end_s - start_s)
        print("Compress: ", end_co - start_co)
        print("Clustering: ", end_c - start_c)

results.close()
rt.finalize()
