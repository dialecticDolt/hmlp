from PyGOFMM_Dist import *
from mpi4py import MPI
import time

prt = PyRuntime()
comm = MPI.COMM_WORLD
prt.init_with_MPI(comm)
nprocs = comm.Get_size()
rank = comm.Get_rank()

m_per = 2000
m = m_per*nprocs
d = 3
print_rank = 1

def pprint(s,comm,prank = print_rank):
    comm.Barrier()
    if(rank==prank):
        print(s)
    comm.Barrier()

def printOwnership(s,comm,prank = print_rank):
    comm.Barrier()
    if(rank==prank):
        local_rids = s.getRIDS()
        print("printing for rank ",rank)
        for i in range(len(local_rids)):
            #NOTE: accessing an element of DistData<RIDS, STAR, T> is based on the RID not the row
            print("RID: ", local_rids[i], " Data: ", s[local_rids[i], 0])
    #else:
        #print(prank)
        #print(rank)
    comm.Barrier()

test = np.asfortranarray(np.random.randn(d, m_per).astype('float32'))

source = PyDistData_CBLK(comm, d, m)
source.rand(0, 1)

pprint("Created Randomized Points")

ks = PyKernel("GAUSSIAN")
pprint("Created PyKernel")

pK = PyDistKernelMatrix(comm, ks, source)
pprint("Created PyKernelMatrix")

pT = PyTreeKM(comm)
pprint("Created PyTree")

conf = PyConfig("GEOMETRY_DISTANCE", m, 128, 64, 128, 0.0001, 0.05, True)
pprint("Created PyConfig Object")

pprint("Starting Compress")
pT.compress(comm, pK, config=conf)
pprint("Completed Compress")

pprint("Testing Conversion of Treelist GIDS to numpy list")
gids = pT.getGIDS()
pprint("The per core length of the Treelist->GID vector is: ")
pprint(len(gids))


#Work with a smaller example to show conversion from block contiguous to treelist
m_per = 20
m = m_per*nprocs
c_data = np.arange(start=d*rank*m_per, stop=d*(rank+1)*m_per).astype('float32')+0.25


pprint("Testing organization of RIDS DistData",comm)
localdata = np.asfortranarray(c_data)
localrids = np.asfortranarray(np.arange(start=rank*m_per, stop=(rank+1)*m_per).astype('int32'))
pprint("Orig locals ",comm)
pprint(localrids,comm)
RIDS_contiguous = PyDistData_RIDS(comm, m,d, arr=localdata, iset=localrids)

pprint("Ownership of continuous",comm)
printOwnership(RIDS_contiguous,comm)

np.random.seed(3)
globalrids = np.asfortranarray(np.arange(start=0, stop = m).astype('int32'))
np.random.shuffle(globalrids)
localrids = globalrids[(rank*m_per):(rank+1)*m_per]

pprint(" ",comm)
pprint("New locals",comm)
pprint(localrids,comm)
pprint(" ",comm)

RIDS_arbitrary = PyDistData_RIDS(comm, m,d, iset=localrids)

RIDS_arbitrary.redistribute(RIDS_contiguous)
pprint("Ownership of arbitrary",comm)
printOwnership(RIDS_arbitrary,comm)

prt.finalize()


