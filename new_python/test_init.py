#from PyRuntime import *
#from PyMatrix import *
from PyGOFMM import *
from mpi4py import MPI

prt = PyRuntime()
prt.init()


print('Initiated!')

# initialize matrix
pK = PySPDMatrix( 5000,5000)

# randomize
pK.randspd(0.0,1.0)

# initalize tree
pT = PyTreeSPD()


# compress matrix
#pT.Compress(pK,1e-2,0.01,128,64,32)




prt.finalize()
