import PyGOFMM_Dist as PyGOFMM
import numpy as np
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import slepc4py
from slepc4py import SLEPc


class GOFMM_Kernel(object):
    
    def __init__(self, comm, N, d, sources, targets=None, kstring="GAUSSIAN", config=None, petsc=True):
        self.rt = PyGOFMM.PyRuntime()
        self.rt.init_with_MPI(comm)
        self.comm = comm
        if not petsc:
            self.K = PyGOFMM.KernelMatrix(comm, sources, targets, kstring, config)
        self.size = N
        if petsc:
            if(targets==None):
                with sources as src:
                    GOFMM_src = PyGOFMM.PyDistData_CBLK(comm, d, N, arr=src.astype('float32'))
                    self.K = PyGOFMM.KernelMatrix(comm, GOFMM_src, targets, kstring, config)
            else:
                with sources as src:
                    with targets as trg:
                        GOFMM_src = PyGOFMM.PyDistData_CBLK(comm, d, N, arr=src.astype('float32'))
                        GOFMM_trg = PyGOFMM.PyDistData_CBLK(comm, d, N, arr=trg.astype('float32'))
                        self.K = PyGOFMM.KernelMatrix(comm, GOFMM_src, GOFMM_trg, kstring, config)
        self.K.compress()

    def __dealloc__(self):
        self.rt.finalize()

    def mult(self, mat, X, Y):
        with X as x:
            with Y as y:
                n = len(x)
                nrhs = 1
                GOFMM_x = PyGOFMM.PyDistData_RIDS(self.comm, m=self.size, n=nrhs, tree=self.K.getTree(), arr=x.astype('float32'))
                GOFMM_y = self.K.evaluate(GOFMM_x)
                for i in range(len(y)):
                    y[i] = GOFMM_y[GOFMM_y.getRIDS()[i], 0]
        return Y

    def getDiagonal(self, mat, result):
        local_gids = self.K.getTree().getGIDS()
        print(local_gids)
        y = np.empty([self.size, 1], dtype='float32')
        with result as y:
            for i in range(len(local_gids)):
                y[i] = self.K.getValue(local_gids[i], local_gids[i])
        return result

    def solve(self, mat, B, X):
        with X as x:
            with B as b:
                n = len(b)
                nrhs = 1
                GOFMM_b = PyGOFMM.PyData(m=n, n=nrhs, arr=x.astype('float32'))
                self.K.solve(GOFMM_b, 1)
                for i in range(len(x)):
                    x[i] = GOFMM_b[i, 0]
                        

petsc4py.init(comm=MPI.COMM_WORLD)
nprocs = MPI.COMM_WORLD.Get_size()

N_per = 2000
N = N_per*nprocs
d = 3

conf = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.0001, 0.01, False)
comm2 = PETSc.COMM_WORLD
comm = MPI.COMM_WORLD

test_points = np.asfortranarray(np.random.randn(d, N_per).astype('float32'))

#test initialization with numpy hmlp
#sources = PyGOFMM.PyDistData_CBLK(comm, d, N, darr=test_points)

#test initialization with petsc
sources = PETSc.Vec().createWithArray(test_points)
with sources as src:
    print(src)
gofmm = GOFMM_Kernel(comm, N, d, sources, config=conf)

A = PETSc.Mat().createPython( [N, N], comm=comm2)
A.setPythonContext(gofmm)
A.setUp()

x, b = A.createVecs()
x.set(1)
with x as x_arr:
    print(x_arr)
b = A*x
with b as b_arr:
    print(b_arr)

A.solve(b, x)

with x as x_arr:
    print(x_arr)
#x.set(1)
#with x as x_arr:
#    print(x_arr)
#b = A*(2*x)
#with b as b_arr:
#    print(b_arr)

#b = A.getDiagonal()
#with b as b_arr:
#    print(b_arr)
