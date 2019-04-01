from PyGOFMM_Dist import *
from mpi4py import MPI
import time

prt = PyRuntime()
comm = MPI.COMM_WORLD
prt.init_with_MPI(comm)
nprocs = comm.Get_size()
rank = comm.Get_rank()

m_per = 20
m = m_per*nprocs
d = 3
print_rank = 1

def pprint(s,pcomm=comm,prank = print_rank):
    pcomm.Barrier()
    if(rank==prank):
        print(s)
    pcomm.Barrier()

def printOwnership(s,pcomm=comm,prank = print_rank):
    pcomm.Barrier()
    if(rank==prank):
        local_rids = s.getRIDS()
        print("printing for rank ",rank)
        for i in range(len(local_rids)):
            #NOTE: accessing an element of DistData<RIDS, STAR, T> is based on the RID not the row
            print("RID: ", local_rids[i], " Data: ", s[local_rids[i], 0])
    #else:
        #print(prank)
        #print(rank)
    pcomm.Barrier()

#test = np.asfortranarray(np.random.randn(d, m_per).astype('float32'))
#
#source = PyDistData_CBLK(comm, d, m)
#source.rand(0, 1)
#
#pprint("Created Randomized Points")
#
#ks = PyKernel("GAUSSIAN")
#pprint("Created PyKernel")
#
#pK = PyDistKernelMatrix(comm, ks, source)
#pprint("Created PyKernelMatrix")
#
#pT = PyTreeKM(comm)
#pprint("Created PyTree")
#
#conf = PyConfig("GEOMETRY_DISTANCE", m, 128, 64, 128, 0.0001, 0.05, True)
#pprint("Created PyConfig Object")
#
#pprint("Starting Compress")
#pT.compress(comm, pK, config=conf)
#pprint("Completed Compress")
#
#pprint("Testing Conversion of Treelist GIDS to numpy list")
#gids = pT.getGIDS()
#pprint("The per core length of the Treelist->GID vector is: ")
#pprint(len(gids))


#Work with a smaller example to show conversion from block contiguous to treelist
c_data = np.arange(start=d*rank*m_per, stop=d*(rank+1)*m_per).astype('float32')+0.25
localdata = np.asfortranarray(np.reshape(c_data,(m_per,d),order='F'))

#pprint(" -- original data (np) --")
#pprint(localdata[0,0])
#pprint(localdata[0,1])
#pprint(localdata[1,0])
#pprint(localdata[1,1])

pprint("Testing organization of RIDS DistData",comm)
localrids = np.asfortranarray(np.arange(start=rank*m_per, stop=(rank+1)*m_per).astype('int32'))
cont_rids = localrids
pprint("Orig locals ",comm)
pprint(localrids,comm)
RIDS_contiguous = PyDistData_RIDS(comm, m,d, darr=localdata, iset=localrids)


pprint("Ownership of contiguous",comm)
printOwnership(RIDS_contiguous,comm)
#pprint(" -- RIDS cont obj  --")
#pprint(RIDS_contiguous[localrids[0],0])
#pprint(RIDS_contiguous[localrids[0],1])
#pprint(RIDS_contiguous[localrids[1],0])
#pprint(RIDS_contiguous[localrids[1],1])

np.random.seed(3)
globalrids = np.asfortranarray(np.arange(start=0, stop = m).astype('int32'))
np.random.shuffle(globalrids)
localrids = globalrids[(rank*m_per):(rank+1)*m_per]
#pprint(" ",comm)
pprint("New locals",comm)
pprint(localrids,comm)
#pprint(" ",comm)

RIDS_arbitrary = PyDistData_RIDS(comm, m,d, iset=localrids)

RIDS_arbitrary.redistribute(RIDS_contiguous)
pprint("Ownership of arbitrary",comm)
printOwnership(RIDS_arbitrary,comm)
#pprint(" -- RIDS arb obj  --")
#pprint(RIDS_arbitrary[localrids[0],0])
#pprint(RIDS_arbitrary[localrids[0],1])
#pprint(RIDS_arbitrary[localrids[1],0])
#pprint(RIDS_arbitrary[localrids[1],1])
#pprint(RIDS_arbitrary[localrids[d],0])
#pprint(RIDS_arbitrary[localrids[d],1])



# testing to array functionality
#bla = RIDS_arbitrary.toArray()
#
#pprint(" -- np out obj  --")
#pprint(bla[0,0])
#pprint(bla[0,1])
#pprint(bla[1,0])
#pprint(bla[1,1])


# test updateRIDS
RIDS_arbitrary.updateRIDS(cont_rids)
pprint("Ownership updated")
printOwnership(RIDS_arbitrary,comm)


prt.finalize()


