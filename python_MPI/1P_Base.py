import PyGOFMM_Dist as PyGOFMM
import numpy as np
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc

class GOFMM_Kernel(object):
    
    def __init__(self, comm, N, d, sources, targets=None, kstring="GAUSSIAN", config=None, petsc=True):
        self.rt = PyGOFMM.PyRuntime()
        self.rt.init_with_MPI(comm)
        self.comm = comm
        if not petsc:
            self.K = PyGOFMM.KernelMatrix(comm, sources, targets, kstring, config, 1)
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

    def getSize(self):
        return self.size

    def getMPIComm(self):
        return self.comm

    #This deep copies the data three times:
    #Once from PETSC -> DistData (could fix with new constructor)
    #Once from stack -> heap in Chenhans code  (could fix by changing hmlp::Evaluate)
    #Once from DistData -> PETSC (to prevent deallocation when GOFMM_y goes out of scope) (could fix if we change hmlp::Evaluate to take in GOFMM_y and overwrite)
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

    def mult_hmlp(self, GOFMM_x):
        GOFMM_y = self.K.evaluate(GOFMM_x)
        return GOFMM_y

    def solve_hmlp(self, GOFMM_b, l):
        GOFMM_x = GOFMM_b #TODO: Placeholder, replace this with a deep copy
        self.K.solve(GOFMM_b, l)

    def getGIDS(self):
        gids = self.K.getTree().getGIDS()
        return gids

    def redistribute_hmlp(self, GOFMM_x):
        n = GOFMM_x.rows()
        d = GOFMM_x.cols()
        GOFMM_y = PyGOFMM.PyDistData_RIDS(self.comm, n, d, tree=self.K.getTree())
        GOFMM_y.redistribute(GOFMM_x)
        return GOFMM_y

    def redistribute(self, X, d, n, nper, form="petsc", transpose=True):
        with X as x:
            if(transpose):
                x = x.reshape(nper, d, order='C')
            x = np.asfortranarray(x)
            GOFMM_x = PyGOFMM.PyDistData_RIDS(self.comm, n, d, darr=x.astype('float32'))
            GOFMM_y = self.redistribute_hmlp(GOFMM_x)
            if(form=="petsc"):
                return PETSc.Vec().createWithArray(np.copy(GOFMM_y.toArray())) #TODO: Fix toArray()
            elif(form=="hmlp"):
                return GOFMM_y
            else:
                raise Exception("Redistribute must return either a PETSc Vec() or HMLP DistData<RIDS, STAR, T>")
            
    
    def getDiagonal(self, mat, result):
        local_gids = self.K.getTree().getGIDS()
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
rank = MPI.COMM_WORLD.Get_rank()

N_per = (int)(4000/nprocs)
N = N_per*nprocs
d = 3
conf = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.0001, 0.01, True)
comm_petsc = PETSc.COMM_WORLD
comm_mpi = MPI.COMM_WORLD
np.random.seed(10)

#Set up artificial points (two classes) that are the same no matter how many processors
class_1 = np.random.randn(d, (int)(np.floor(N/2)))+5
class_2 = np.random.randn(d, (int)(np.ceil(N/2)))
test_points = np.concatenate((class_1, class_2), axis=1) #data points shape = (d, N_per)

offset = 0
if rank==0:
    test_points = test_points[:, rank*N_per:(rank+1)*N_per+offset]
else:
    test_points = test_points[:, rank*N_per+offset:(rank+1)*N_per+offset]

N_per = test_points.shape[1]
true_classes = np.ones([1, test_points.shape[1]]) #class vector (pi)

classes = np.ones([1, test_points.shape[1]])
for i in range(test_points.shape[1]):
    if(np.linalg.norm(test_points[:, i])>=2):
        true_classes[0, i] = 2
    if(np.random.rand() > 0.5):
        classes[0, i] = 2
    #if(np.random.rand() < 0.25):
    #    classes[0, i] = 3;

test_points = np.asfortranarray(test_points.astype('float32'))
source_points = PETSc.Vec().createWithArray(test_points)
true_classes = PETSc.Vec().createWithArray(true_classes)
classes = PETSc.Vec().createWithArray(classes)


gofmm = GOFMM_Kernel(comm_mpi, N, d, source_points, config=conf) #set up python context for gofmm
redistributed_points = gofmm.redistribute(source_points, d, N, nper=N_per, form='hmlp') #redistribute to HMLP Dist Data 
redistributed_true_classes = gofmm.redistribute(true_classes, 1, N, nper=N_per, form='hmlp')
redistributed_classes = gofmm.redistribute(classes, 1, N, nper=N_per, form="hmlp")

#set up gofmm operator
A = PETSc.Mat().createPython( [N, N], comm=comm_petsc)
A.setPythonContext(gofmm)
A.setUp()

#Renaming for convenience
classes = redistributed_classes
points = redistributed_points
true_classes = redistributed_true_classes

rids = classes.getRIDS()
local_rows = len(rids)
problem_size = A.getSize()[0]

local_ones = np.ones([local_rows, 1], dtype='float32', order='F')
Ones = PyGOFMM.PyDistData_RIDS(comm_mpi, problem_size, 1, iset=rids.astype('int32'), darr=local_ones)

D = gofmm.mult_hmlp(Ones)

np.savetxt('ReferenceD_'+str(rank), D.toArray())
np.savetxt('Referencerids_'+str(rank), rids)


