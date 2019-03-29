import PyGOFMM_Dist as PyGOFMM
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print_rank = 0

#Start HMLP Runtime
rt = PyGOFMM.PyRuntime()
rt.init_with_MPI(comm)

#Setup GOFMM parameters

n_local = 2000
n = n_local*size

ktype = "GAUSSIAN"
d = 20  #NOTE: SGEMM Error when d=10
m = 128     #64
k = 64      #32
s = 128     #32
stol = 1e-4 #1e-5
budget = 0.01
sec_acc = True #NOTE: SGEMM Error when sec_acc=True

config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", n, m, k, s, stol, budget, sec_acc)

#Create points for kernel matrix
np.random.seed(rank*10)
source_points = np.random.rand(d, n_local)
source_points = np.asarray(source_points, dtype='float32', order='F')

source = PyGOFMM.PyDistData_CBLK(comm, d, n, darr=source_points)

#Create kernel matrix
K = PyGOFMM.KernelMatrix(comm, source, kstring="GAUSSIAN", conf=config, bandwidth=1)

#Compress kernel matrix
K.compress()

#Create RHS
#(Done seperately to show consecutive calls of evaluate)

#Ones on Rank 0, Zeros elsewhere
if rank==0:
    local_test1 = np.ones([n_local, 1], dtype='float32', order='F')
else:
    local_test1 = np.zeros([n_local, 1], dtype='float32', order='F')

R0Ones = PyGOFMM.PyDistData_RIDS(comm, n, 1, tree=K.getTree(), darr=local_test1)

#Zeros on Rank 0, Ones elsewhere
if rank==1:
    local_test2 = np.ones([n_local, 1], dtype='float32', order='F')
else:
    local_test2 = np.zeros([n_local, 1], dtype='float32', order='F')

R0Zeros = PyGOFMM.PyDistData_RIDS(comm, n, 1, tree=K.getTree(), darr=local_test2)


#All Ones
local_ones = np.ones([n_local, 1], dtype='float32', order='F')
Ones = PyGOFMM.PyDistData_RIDS(comm, n, 1, tree=K.getTree(), darr = local_ones)

#Run Evaluate

R0Ones_Result = K.evaluate(R0Ones)
R0Zeros_Result = K.evaluate(R0Zeros)
Ones_Result = K.evaluate(Ones)

#Show output
rids = R0Ones_Result.getRIDS() #get GOFMM ordering (defined by compression tree i.e. same across all evaluate calls)

#First element of each, third should be a sum of the first two up to precision of matvec
print("Rank: ", rank, " RID: ", rids[0])
print("Rank: ", rank, " ", R0Ones_Result[rids[0], 0])
print("Rank: ", rank, " ", R0Zeros_Result[rids[0], 0])
print("Rank: ", rank, " ", Ones_Result[rids[0], 0])
print(" ")


#Print GIDS of Tree (just to check against RIDS)
gids = K.getTree().getGIDS()
print("Rank: ", rank, "GIDS Length: ", len(gids), " GIDS: ", gids)

 
