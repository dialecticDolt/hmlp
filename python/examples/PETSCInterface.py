import pygofmm.core as PyGOFMM
import pygofmm.mltools as MLTools
import pygofmm.mltools.algorithms as FMML
from pygofmm.petsc import *

from mpi4py import MPI

import petsc4py
import petsc4py as petsc
import slepc4py as slepc

import numpy as np
import argparse
import sys


comm = MPI.Comm.Clone(MPI.COMM_WORLD)
nprocs = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Perform a matvec and solve using PETSC datastructures')

parser.add_argument('-N', type=int, required=False, default= 10000, help="Specify the size of the random dataset. Set --scaling strong or weak")

parser.add_argument('--scaling', type=str, dest='scaling', choices=['strong', 'weak'],default='weak', help="Scaling of the point set size. weak = N, strong=N*p")

parser.add_argument('-d', type=int, required=False, default = 5, help="Specify the dimension of the random point set")

parser.add_argument('-leaf', type=int, required=False, default = 128, help="Specify the leaf size of the kd-tree")

parser.add_argument('-k', type=int, required=False, default = 64, help="Specify the number of nearest neighbors to use in determining the far field and sparse corrections")

parser.add_argument('-max_rank', type=int, required=False, default=128, help="Specify the maximum rank of the internal nodes")

parser.add_argument("-tolerance", type=float, required=False, default = 1e-3, help="Specify the accuracy tolarance of the internal low-rank approximations")

parser.add_argument("-budget", type=float, required=False, default = 0, help="Specify the budget of the exact evaulation close neighbor list")

args = parser.parse_args()

leaf_node_size = args.leaf
neighbors = args.k
maximum_rank = args.max_rank
tol = args.tolerance
budget = args.budget

rt = PyGOFMM.Runtime()
rt.initialize(comm)

if args.scaling=='strong':
    N = args.N*nprocs
elif args.scaling=='weak':
    N = args.N

d = args.d

#Construct the point set
#np.random.seed(10)
nclasses = 2
extra = (N % nclasses)
for i in range(nclasses):
    offset = np.random.rand(d, 1)*10
    print(offset)
    if(extra):
        split = 1
        extra= extra -1
    else:
        split = 0
    new_class = np.random.randn(d, (int)(np.floor(N/nclasses)) + split) + offset
    if i == 0:
        point_set = new_class
    else:
        point_set = np.concatenate((point_set, new_class), axis=1)

point_set = PyGOFMM.reformat(point_set)

#Redistribute points to cyclic partition
start_s = MPI.Wtime()
sources, gids_owned = PyGOFMM.distribute_cblk(comm, point_set)


#Setup and compress the kernel matrix
setup_time = MPI.Wtime()
config = PyGOFMM.Config("GEOMETRY_DISTANCE", N, leaf_node_size, neighbors, maximum_rank, tol, budget, False)
K = PyGOFMM.KernelMatrix(comm, sources, config=config, bandwidth=1.3)
setup_time = MPI.Wtime() - setup_time

compress_time = MPI.Wtime()
K.compress()
compress_time = MPI.Wtime() - compress_time

##Setup Complete
comm = K.get_comm()
petsc4py.init(comm=comm)


#Create Wrapper for Kernel Object
#normalize=False -> matvec is K*x
#normalize=True  -> matvec is (D^-1 K) * x where D is the degree matrix diag(K*1)
kernel_wrapper = Kernel_Handler(K, normalize=False, regularization=1.0)

#Create PETSC Python Context
A = PETSC_Handler(kernel_wrapper, comm)

x, b = A.createVecs()
x.set(1.0)

b = A*x

#b = A.solve(x)

rt.finalize()
