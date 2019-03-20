from PyGOFMM_Dist import *
from mpi4py import MPI
import time

prt = PyRuntime()
prt.init_with_MPI(MPI.COMM_WORLD)
nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
print("THIS IS MY RANK")
print(rank)

def pprint(stuff):
    if(rank==0):
        print(stuff)

def pprint2(stuff):
    if(rank==1):
        print(stuff[1, 0])

m_per = 2000
m = m_per*nprocs
d = 2

test = np.asfortranarray(np.random.randn(d, m_per).astype('float32'))
source = PyDistData_CBLK(MPI.COMM_WORLD, d, m, darr=test)
pprint("Created Randomized Points")

ks = PyKernel("GAUSSIAN")
ks.setBandwidth(0.00001)
pprint("Created PyKernel")

pK = PyDistKernelMatrix(MPI.COMM_WORLD, ks, source)
pprint("Created PyKernelMatrix")

pT = PyTreeKM(MPI.COMM_WORLD)
pprint("Created PyTree")

#print("Testing kernel values...")
#for i in range(1, 10):
#    print(pK.getvalue(i, 10))
#print("...Finished Printing kernel values")

conf = PyConfig("GEOMETRY_DISTANCE", m, 128, 64, 128, 0.0001, 0.05, True)
pprint("Created PyConfig Object")

pprint("Starting Compress")
pT.compress(MPI.COMM_WORLD, pK, config=conf)
pprint("Completed Compress")


nrhs = 10;

pprint("Create Weights Object")
pprint("Test conversion from RBLK to RIDS")

weights = PyDistData_RBLK(MPI.COMM_WORLD, m, nrhs);
weights.randn()

emptyRIDS = PyDistData_RIDS(MPI.COMM_WORLD, m=m, n=nrhs, tree=pT);
emptyRIDS.loadRBLK(weights)

pprint("Test Evaluate with RBLK weights")
pprint2(weights)
result = pT.evaluateRBLK(weights)
pprint2(result)

pprint("Test Conversion from RIDS to RBLK")
weights = PyDistData_RIDS(MPI.COMM_WORLD, m=m, n=nrhs, tree=pT);
weights.randn()

emptyRBLK = PyDistData_RBLK(MPI.COMM_WORLD, m=m, n=nrhs);
emptyRBLK.loadRIDS(weights)

pprint("Test Evaluate with RIDS weights")
pT.evaluateRIDS(weights)

pprint("Finished Evaluate")

pprint("Factorize")
y = PyData(m_per, nrhs)
y.randn(0, 1)
pT.factorize(1)
pprint("Solve")

pprint2(y)
pT.solve(y)
pprint2(y)

#pprint("Starting Cleanup")
#del pT
#del pK
#del ks

prt.finalize()


