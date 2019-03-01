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
print('Formed matrix!')

# initalize tree
pT = PyTreeSPD()

# compress matrix
pT.PyCompress(pK,1e-5,0.01,64,8,64,False)
print('Formed compressed tree!')

# evaluate
ww = PyData(5000,1)
ww.rand(0.0,1.0)
uu = pT.PyEvaluate(ww)
print(uu.col())
print(uu.row())

print('Evaluated!')


# factorize
pT.PyFactorize(1.0)
print('Factorized!')


# solve
ww_est = uu.MakeCopy()
pT.PySolve(ww_est)

#print(ww.getvalue(1,0))
#print(ww_est.getvalue(1,0))
#ww_est.setvalue(1,0,5.0)
#
#print(ww.getvalue(1,0))
#print(ww_est.getvalue(1,0))
print('Solved!')


prt.finalize()
