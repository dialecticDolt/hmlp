import PyGOFMM_Dist as PyGOFMM
import numpy as np
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import matplotlib.pyplot as plt

def getCBLKOwnership(N, rank, nprocs):
    cblk_idx = np.asarray(np.arange(rank, N, nprocs), dtype='int32', order='F')
    return cblk_idx

class GOFMM_Kernel(object):
    def __init__(self, comm, N, d, sources, targets=None, kstring="GAUSSIAN", config=None, bwidth=1.0):
        self.comm = comm
        self.rt = PyGOFMM.PyRuntime()
        self.rt.init_with_MPI(comm)
        self.K = PyGOFMM.KernelMatrix(comm, sources, targets, kstring, config, bwidth)
        self.K.compress()

    def mult_hmlp(self, GOFMM_x):
        GOFMM_y = self.K.evaluate(GOFMM_x)
        return GOFMM_y


    def getGIDS(self):
        gids = self.K.getTree().getGIDS()
        return gids


    def redistribute_hmlp(self, GOFMM_x, rids=None):
        n = GOFMM_x.rows()
        d = GOFMM_x.cols()
        if rids is None:
            GOFMM_y = PyGOFMM.PyDistData_RIDS(self.comm, n, d, tree=self.K.getTree())
        else:
            GOFMM_y = PyGOFMM.PyDistData_RIDS(self.comm, n, d, iset=rids)
        GOFMM_y.redistribute(GOFMM_x)
        return GOFMM_y

    def getMPIComm(self):
        return self.comm

def KMeansHelper(gofmm, GOFMM_classes, D, nclasses):
    comm = gofmm.getMPIComm()
    rank = comm.Get_rank()
    size = comm.Get_size()
    rids = GOFMM_classes.getRIDS()
    local_rows = len(rids)
    N = D.rows()

    #setup indicator vectors with current classes
    local_H = np.zeros([local_rows, nclasses], dtype='float32', order='F')
    for i in range(local_rows):
        local_H[ i, (int)( GOFMM_classes[rids[i], 0]-1 ) ] = 1.0

    H = PyGOFMM.PyDistData_RIDS(comm, N, nclasses, iset=rids.astype('int32'), darr=local_H)

    #multiply by kernel matrix
    KH = gofmm.mult_hmlp(H)

    #setup lookup matrices
    #Generate HKH, HDH, DKH
    HKH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    HDH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    DKH = np.zeros([local_rows, nclasses], dtype='float32', order='F')

    for i in range(nclasses):
        for j in range(nclasses):
            for r in rids:
                HKH_local[i, j] += H[r, j] * KH[r, i]
                HDH_local[i, j] += H[r, j] * D[r, 0] * H[r, i]

    for i in range(local_rows):
        for j in range(nclasses):
            DKH[i, j] = 1/D[rids[i], 0] * KH[rids[i], j]

    HKH = np.zeros(HKH_local.shape, dtype='float32', order='F')
    HDH = np.zeros(HDH_local.shape, dtype='float32', order='F')
    comm.Allreduce(HKH_local, HKH, op=MPI.SUM)
    comm.Allreduce(HDH_local, HDH, op=MPI.SUM)
    return (DKH, HKH, HDH)





#TEST SCRIPT

petsc4py.init(comm=MPI.COMM_WORLD)
nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

N = 10000
d = 3

conf = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.001, 0.01, False)
comm_petsc = PETSc.COMM_WORLD
comm_mpi = MPI.COMM_WORLD
np.random.seed(10)

#Set up artificial points (two classes)

#local copy of all points
class_1 = np.random.randn(d, (int)(np.floor(N/2)))+2
class_2 = np.random.randn(d, (int)(np.ceil(N/2)))
test_points = np.concatenate((class_1, class_2), axis=1) #data points shape = (d, N_per)

#get owned points in CBLK ()cyclic) ordering
GIDs_Owned = getCBLKOwnership(N, rank, nprocs)
CBLK_points = test_points[:, GIDs_Owned]
CBLK_points = np.asarray(CBLK_points, dtype='float32', order='F')

#Set up class information
n_local = len(GIDs_Owned)
true_classes = np.ones(n_local, dtype='float32', order='F')
classes = np.ones(n_local, dtype='float32', order='F')

i = 0
for point in CBLK_points.T:
    if( np.linalg.norm(point)>4 ):
        true_classes[i] = 2
    if( GIDs_Owned[i]%3==0 ):
        classes[i] = 2
    if( GIDs_Owned[i]%5==0 ):
        classes[i] = 3
    i = i+1

#create gofmm context
source_points = PyGOFMM.PyDistData_CBLK(comm_mpi, m=d, n=N, darr=CBLK_points)
gofmm = GOFMM_Kernel(comm_mpi, N, d, source_points, config = conf)


#redistribute classes to treelist ordering (from CBLK cyclic)
GOFMM_classes = PyGOFMM.PyDistData_RIDS(comm_mpi, m=N, n=1, iset=GIDs_Owned, arr=classes)

GOFMM_classes = gofmm.redistribute_hmlp(GOFMM_classes)
rids = GOFMM_classes.getRIDS()

##start Kernel K-Means
k = 3

#compute normalizing matrix D
local_ones = np.ones([n_local, 1], dtype='float32', order='F')
Ones = PyGOFMM.PyDistData_RIDS(comm_mpi, N, 1, iset=rids.astype('int32'), darr=local_ones)
D = gofmm.mult_hmlp(Ones)

#start main loop
maxiter = 10
for i in range(maxiter):
    (DKH, HKH, HDH) = KMeansHelper(gofmm, GOFMM_classes, D, k)
    Diag = np.ones([n_local, 1])

    #compute similarity
    Similarity = np.zeros([n_local, k], dtype='float32', order='F')
    for i in range(n_local):
        for p in range(k):
            Similarity[i, p] = Diag[i]/(D[rids[i], 0]*D[rids[i], 0]) - 2 * DKH[i, p]/HDH[p, p] + HKH[p, p]/(HDH[p, p] * HDH[p, p])

    #update class information
    for i in range(n_local):
        GOFMM_classes[rids[i], 0] = np.argmin(Similarity[i, :])+1


GOFMM_classes = gofmm.redistribute_hmlp(GOFMM_classes, GIDs_Owned)
np_classes = GOFMM_classes.toArray()
comm = comm_mpi

recvbuf = comm.allgather(np_classes)
np_classes = np.concatenate(recvbuf, axis=0).astype('int32')

recvbuf = comm.allgather(GIDs_Owned)
gids = np.concatenate(recvbuf, axis=0).astype('int32')

points = test_points[:, gids].T

plt.scatter(points[:, 0], points[:, 1], c=np_classes.flatten())
plt.show()


