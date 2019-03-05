from PyGOFMM import *
from mpi4py import MPI

prt = PyRuntime()
prt.init()

print('Initiated!')

m = 2000
n = 2000
d = 20

source = PyData(d, m)
source.rand(0.0, 1.0)

target = PyData(d, n)
target.rand(0.0, 1.0)

ks = PyKernel("GAUSSIAN")
ks.SetBandwidth(1.0)

pK = PyKernelMatrix(source, ks)

pT = PyTreeKM()

print(pK.getvalue(10, 10))

conf = PyConfig("GEOMETRY_DISTANCE", 2000, 128, 64, 32, 0.001, 0.01, True)

pT.PyCompress(pK, config=conf)
prt.finalize()


