import pygofmm.core as PyGOFMM
from mpi4py import MPI
import numpy as np
import argparse
import sys

comm = MPI.Comm.Clone(MPI.COMM_WORLD)
nprocs = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Compute approximatie all nearest neighbors')

parser.add_argument('-file', type=str, required=False, default=None,
        help="Specify the file to load")
parser.add_argument('-N', type=int, required=False, default= 10000, help="Specify the size of the random dataset. Set --scaling strong or weak")
parser.add_argument('-leaf', type=int, required=False, default = 128, help="Specify the leaf size of the kd-tree for nearest neighbor approximation")
parser.add_argument('-k', type=int, required=False, default=64, help="Specify the number of neighbors to search for")
parser.add_argument('-d', type=int, required=False, default = 5, help="Specify the dimension of the random point set")
parser.add_argument('--scaling', dest='scaling', choices=['strong', 'weak'],default='weak', help="Scaling of the point set size. weak = N*p, strong=N")

args = parser.parse_args()

rt = PyGOFMM.Runtime()
rt.initialize(comm)

k = args.k

if args.file:
    #Option 1: Load a data set
    N = 10
else:
    #Option 2: Construct a random point set
    #Number of points in dataset. Setup for strong scaling
    if args.scaling == 'strong':
        N = args.N
    elif args.scaling == 'weak':
        N = args.N*nprocs

    d = args.d

    np.random.seed(10)
    class1 = np.random.randn(d, (int)(np.floor(N/2))) +5
    class2 = np.random.randn(d, (int)(np.floor(N/2)))
    point_set = np.concatenate((class1, class2), axis=1)
    point_set = PyGOFMM.reformat(point_set)

if(rank == 0):
    print("Beginning Nearest Neighbor Search with ", N, " points")
    sys.stdout.flush()

nearest_neighbor_list = PyGOFMM.All_Nearest_Neighbors(comm, N, k, point_set, leafnode=args.leaf)
distances, gids = nearest_neighbor_list.to_numpy()
#These are k x N indexed on gid in the second parameter
#Distribution is block cyclic row. Processor i owns gid % nprocs == i

gid = 2
if(rank == gid % nprocs):
    print("Printing ", k, " Nearest Neighbors for GID: ", gid)
    #The 0th element is the point itself, start at 1 for neighbors
    for i in range(0, k):
        print("ID: ", gids[i, gid], " Distance = ", distances[i, gid])

sys.stdout.flush()
rt.finalize()


