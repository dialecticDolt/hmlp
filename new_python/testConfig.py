from PyGOFMM import *
from mpi4py import MPI

prt = PyRuntime()
prt.init()

print('Initiated!')

pK = PySPDMatrix(5000, 5000)

pK.randspd(0.0, 1.0)

pT = PyTreeSPD()

conf = PyConfig("ANGLE_DISTANCE", 5000, 128, 64, 32, 0.001, 0.01, True)

pT.PyCompress(pK, config=conf)
prt.finalize()


