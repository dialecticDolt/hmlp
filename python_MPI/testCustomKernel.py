from PyGOFMM_Dist import *
from mpi4py import MPI
import time

prt = PyRuntime()
prt.init_with_MPI(MPI.COMM_WORLD)
nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

m = 20000
d = 10

#test = np.asfortranarray(np.random.randn(d, m_per).astype('float32'))
source = PyDistData_CBLK(MPI.COMM_WORLD, d, m)
source.randn()

ks = PyKernel("USER_DEFINE")
pK = PyDistKernelMatrix(MPI.COMM_WORLD, ks, source)
pT = PyTreeKM(MPI.COMM_WORLD)

conf = PyConfig("GEOMETRY_DISTANCE", m, 128, 64, 128, 0.00001, 0.01, False)
pT.compress(MPI.COMM_WORLD, pK, config=conf)

nrhs = 10;
wRIDS = PyDistData_RIDS(MPI.COMM_WORLD, m=m, n=nrhs, tree=pT);
wRIDS.randn();

result = pT.evaluateRIDS(wRIDS)

prt.finalize()


