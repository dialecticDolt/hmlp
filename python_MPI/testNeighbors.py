from PyGOFMM_Dist import *
from mpi4py import MPI
import time

prt = PyRuntime()
prt.init_with_MPI(MPI.COMM_WORLD)
nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

m_per = 2000
m = m_per*1
d = 3
k = 3
#test = np.asfortranarray(np.random.randn(d, m_per).astype('float32'))
#source = PyDistData_CBLK(MPI.COMM_WORLD, d, m, darr=test)

source = np.asfortranarray(np.random.randn(d, m_per).astype('float32'))
PyNNList = FindAllNeighbors(MPI.COMM_WORLD, m, k, source)

print(PyNNList.rows())
print(PyNNList.cols())
print(PyNNList[1, 1])

distances, gids = PyNNList.toNumpy()

ID = 2
if(rank==0):
    print("Neighbors for GID = ", ID)
    #The 0th element is the points itself
    for i in range(1, k):
        print("ID = ", gids[i, ID], "  Distance = ", distances[i, ID])

prt.finalize()


