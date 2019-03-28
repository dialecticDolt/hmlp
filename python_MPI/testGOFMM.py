#from PyRuntime import *
#from PyMatrix import *
from PyGOFMM_Dist import *
from mpi4py import MPI
import numpy as np
#import IPython

# mpi stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


prt = PyRuntime()
prt.init_with_MPI(comm)

# parameters
n = 2000 # number of points
d = 10 # dimensionality of data
ktype = 'GAUSSIAN' # kernel type
print_rank = 1 # rank to print from
m = 64
k = 32
s = 32
stol = 1e-5
budget = 0.01

print(print_rank)

if rank == print_rank:
    print('Initiated!')

# initiate sources
Xtr = PyDistData_CBLK(comm,m = d, n = n)
Xtr.rand(0.0,1.0)

# initialize pykernel
kernel = PyKernel(kstring = ktype)
if rank == print_rank:
    print('Data and kernel_s formed!')

# initialize matrix
KK = PyDistKernelMatrix(comm,kernel,Xtr)
if rank == print_rank:
    print('Matrix formed!')

# initalize tree?
pT = PyTreeKM(comm)

# initialize config
conf = PyConfig("GEOMETRY_DISTANCE",n,m,k,s,stol,budget,False)

# compress matrix
pT.compress(comm,KK,config=conf)
if rank == print_rank:
    print('Formed compressed tree!')

# evaluate/compute error
pT.test_error()
#ww = PyData(5000,10)
#uu = pT.PyEvaluate(ww)


prt.finalize()
