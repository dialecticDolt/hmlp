from PyGOFMM import *
from mpi4py import MPI

prt = PyRuntime()
prt.init()

print('Initiated Runtime!')

m = 2000
n = 2000
d = 20

source = PyData(d, m)
source.rand(0.0, 1.0)

target = PyData(d, n)
target.rand(0.0, 1.0)

print("Created Randomized Points")

ks = PyKernel("USER_DEFINE")
ks.SetBandwidth(1.0)
print("Created PyKernel")

pK = PyKernelMatrix(source, ks)
print("Created PyKernelMatrix")

pT = PyTreeKM()
print("Created PyTree")

print("Testing kernel values...")
for i in range(1, 10):
    print(pK.getvalue(i, 10))
print("...Finished Printing kernel values")

conf = PyConfig("GEOMETRY_DISTANCE", 2000, 128, 64, 32, 0.001, 0.01, True)
print("Created PyConfig Object")

print("Starting Compress")
pT.PyCompress(pK, config=conf)
print("Completed Compress")


print("Starting Cleanup")
del pT
del pK
del ks

prt.finalize()


