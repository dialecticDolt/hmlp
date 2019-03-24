from PyGOFMM_Dist import *
from mpi4py import MPI
import time

prt = PyRuntime()
prt.init_with_MPI(MPI.COMM_WORLD)
nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

m_per = 2000
m = m_per*nprocs
d = 2

def pprint(s):
    if(rank==0):
        print(s)

def printOwnership(s):
    if(rank==0):
        local_rids = s.getRIDS()
        for i in range(len(local_rids)):
            #NOTE: accessing an element of DistData<RIDS, STAR, T> is based on the RID not the row
            print("RID: ", local_rids[i], " Data: ", s[local_rids[i], 0])

test = np.asfortranarray(np.random.randn(d, m_per).astype('float32'))

source = PyDistData_CBLK(MPI.COMM_WORLD, d, m)
source.rand(0, 1)

pprint("Created Randomized Points")

ks = PyKernel("GAUSSIAN")
pprint("Created PyKernel")

pK = PyDistKernelMatrix(MPI.COMM_WORLD, ks, source)
pprint("Created PyKernelMatrix")

pT = PyTreeKM(MPI.COMM_WORLD)
pprint("Created PyTree")

conf = PyConfig("GEOMETRY_DISTANCE", m, 128, 64, 128, 0.0001, 0.05, True)
pprint("Created PyConfig Object")

pprint("Starting Compress")
pT.compress(MPI.COMM_WORLD, pK, config=conf)
pprint("Completed Compress")

pprint("Testing Conversion of Treelist GIDS to numpy list")
gids = pT.getGIDS()
pprint("The per core length of the Treelist->GID vector is: ")
pprint(len(gids))


#Work with a smaller example to show conversion from block contiguous to treelist
m_per = 20
m = m_per*nprocs

pprint("Testing organization of RIDS DistData")
localdata = np.asfortranarray(np.arange(start=rank, stop=rank+m_per).astype('float32')+0.25)
localrids = np.asfortranarray(np.arange(start=rank, stop=rank+m_per).astype('int32'))
RIDS_contiguous = PyDistData_RIDS(MPI.COMM_WORLD, d, m, arr=localdata, iset=localrids)

printOwnership(RIDS_contiguous)

globalrids = np.asfortranarray(np.arange(start=0, stop = m).astype('int32'))
np.random.shuffle(globalrids)
localrids = globalrids[rank:rank+m_per]

pprint(" ")
pprint(localrids)
pprint(" ")

RIDS_arbitrary = PyDistData_RIDS(MPI.COMM_WORLD, d, m, iset=localrids)

RIDS_arbitrary.redistribute(RIDS_contiguous)
printOwnership(RIDS_arbitrary)

prt.finalize()


