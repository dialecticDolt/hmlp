import pygofmm.core as PyGOFMM
from mpi4py import MPI
import numpy as np
import argparse
import sys
import time
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

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

def distance(R, Q):
    D = -2*np.dot(Q, R.T)
    Q2 = np.linalg.norm(Q, axis=1)**2
    R2 = np.linalg.norm(R, axis=1)**2
    D = D + Q2[:, np.newaxis]
    D = D+R2
    return D


def knn(gids, R, Q, k):
    N, d = Q.shape
    NL = np.zeros([N, k])
    ND = np.zeros([N, k])
    print(N)
    print(gids.shape)
    dist = distance(R, Q)
    for q_idx in range(N):

        lids = np.argpartition(dist[q_idx, ...], k)[:k]
        ND[q_idx, ...] = dist[q_idx, lids]
        tgids = gids[lids]

        shuffle = np.argsort(ND[q_idx, ...])
        ND[q_idx, ...] = ND[q_idx, shuffle]
        tgids = tgids[shuffle]
        NL[q_idx, ...] = tgids

    return NL, ND


def neighbor_dist(a, b):
    lib = np
    a_list = a[0]
    a_dist = a[1]

    b_list = b[0]
    b_dist = b[1]

    Na, ka = a_list.shape
    Nb, kb = b_list.shape
    print(Na, Nb)
    assert(Na == Nb)
    assert(ka == kb)

    changes = 0
    nn_dist = lib.zeros(Na)
    first_diff = lib.zeros(Na)
    for i in range(Na):
        changes += (ka - lib.sum(a_list[i] == b_list[i]))
        nn_dist[i] = lib.abs(a_dist[i, -1] - b_dist[i, -1])/lib.abs(a_dist[i, -1])
        diff_array = lib.abs(a_list[i, ...] - b_list[i, ...])
        first_diff[i] = lib.argmax(diff_array > 0.5)

    perc = changes/(Na*ka)

    return perc, lib.mean(nn_dist), first_diff


if True:
    N = 2**16
    #Option 1: Load a data set
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    #X = np.random.rand(N, 10)
    X = X[:N, :]
    truth = knn(np.arange(0, N), X[:, :], X[:100, :], k)
    print(truth)

    ext_t = time.time()
    #kdt = KDTree(X, leaf_size=256, metric='euclidean')
    #kdt.query(X, k=64, return_distance=False)
    nbrs = NearestNeighbors(n_neighbors=64, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(distances, indices)
    ext_t = time.time() - ext_t
    print("SCIPY", ext_t)
    raise Exception("STOP")
    X = X[:N, :].T  
    point_set = PyGOFMM.reformat(X)
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

f = open("out", "w")
f.close()
for trees in range(10, 100, 10):
    f = open("out", "a")
    search_t = time.time()
    nearest_neighbor_list = PyGOFMM.All_Nearest_Neighbors(comm, N, k, point_set, leafnode=args.leaf, trees=trees)
    search_t = time.time() - search_t
    print("Search Time", search_t)
    distances, gids = nearest_neighbor_list.to_numpy()
    #These are k x N indexed on gid in the second parameter
    #Distribution is block cyclic row. Processor i owns gid % nprocs == i

    distances, gids = (distances.T, gids.T)
    approx = (gids[:100], distances[:100])
    print(approx)
    print(truth)
    #print(truth[0].shape)
    a = neighbor_dist(truth, approx)
    print(a)
    f.write('\n'+str(trees)+',' + str(a[0]) + ','+str(search_t))
    f.close()
    sys.stdout.flush()

rt.finalize()


