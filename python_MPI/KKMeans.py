import PyGOFMM_Dist as PyGOFMM
import numpy as np
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import slepc4py
from slepc4py import SLEPc
import matplotlib.pyplot as plt

import matplotlib
matplotlib.interactive(False)



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

    def redistributeAny_hmlp(self, GOFMM_x, rids):
        n = GOFMM_x.rows()
        d = GOFMM_x.cols()
        GOFMM_y = PyGOFMM.PyDistData_RIDS(self.comm, n, d, iset=rids)
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
            
    def redistributeAny(self, X, d, n, nper, target_rids, form="petsc", transpose=True):
        with X as x:
            if (transpose):
                x = x.reshape(nper, d, order='C')
            x = np.asfortranarray(x)
            GOFMM_x = PyGOFMM.PyDistData_RIDS(self.comm, n, d, darr=x.astype('float32'))
            GOFMM_y = self.redistributeAny_hmlp(GOFMM_x, target_rids)
            if(form=="petsc"):
                return PETSc.Vec().createWithArray(np.copy(GOFMM_y.toArray()))
            elif(form=="hmlp"):
                return GOFMM_y
            else:
                raise Exception("redistribute must returen either a PETSc Vec() or HMLP DistData<RIDS, STAR, T>")

    

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

def learnMapping(comm, D):
    reference_data = np.loadtxt('D_1p')
    reference_rids = np.loadtxt('rids_1p').astype('int32')

    local_target_rids = D.getRIDS().astype('int32')
    local_target_data = np.asarray(D.toArray(), order='C')

    recvbuf = comm.allgather(local_target_data)
    target_data = np.concatenate(recvbuf, axis=0).astype('float32')

    recvbuf = comm.allgather(local_target_rids)
    target_rids = np.concatenate(recvbuf, axis=0).astype('int32')

    ridsMap = dict()
    index1 = 0
    #This is a hacked together O(N^2) implementation. Not for final use, just to test idea
    for i in reference_data:
        index2 = np.argmin(np.abs(target_data-i))
        ridsMap[target_rids[index2]] = reference_rids[index1]
        index1 += 1
    
    local_reference_rids = np.zeros([1, local_target_rids.size], order='F', dtype='int32')
    i = 0;
    for target_rid in local_target_rids:
        local_reference_rids[0, i] = ridsMap[target_rid]
        i += 1
    
    local_target_rids = np.asfortranarray(local_target_rids)

    return (local_target_rids, local_reference_rids[0])
    


#For standard K Means
def computeCenters(A, GOFMM_points, GOFMM_classes, nclasses=2):
    gofmm = A.getPythonContext()
    comm = gofmm.getMPIComm()
    nprocs = comm.Get_size()
    rids = GOFMM_classes.getRIDS()
    local_rows = len(rids)
    d = GOFMM_points.cols()    

    centroids_local = np.zeros([nclasses, d], dtype='float32', order='F')
    centroids = np.copy(centroids_local)

    counts = np.zeros([nclasses, 1], dtype='float32')
    for r in rids:
        c = (int)(GOFMM_classes[r, 0] - 1)
        for l in range(d):
            centroids_local[c, l] +=  GOFMM_points[r, l]
        counts[c] += 1

    for i in range(nclasses):
        centroids_local[i, :] = centroids_local[i, :]/counts[i]
    
    comm.Allreduce(centroids_local, centroids, op=MPI.SUM)

    return centroids/nprocs

#setup lookup matricies
def KMeansPrep(A, GOFMM_classes, nclasses):
    #generate class indicator matrix H
    gofmm = A.getPythonContext()
    comm = gofmm.getMPIComm()
    size = comm.Get_size()
    rank = comm.Get_size()
    rids = GOFMM_classes.getRIDS()
    local_rows = len(rids)
    problem_size = A.getSize()[0]
    local_H = np.zeros([local_rows, nclasses], dtype='float32', order='F')
    for i in range(local_rows):
        local_H[i, (int)(GOFMM_classes[rids[i], 0]-1) ] = 1.0

   # H = PyGOFMM.PyDistData_RIDS(comm, problem_size, nclasses, iset=rids.astype('int32'), darr=np.asfortranarray(np.transpose(local_H)))
    H = PyGOFMM.PyDistData_RIDS(comm, problem_size, nclasses, iset=rids.astype('int32'), darr=local_H)

    #generate vector of ones
    local_ones = np.ones([local_rows, 1], dtype='float32', order='F')
    Ones = PyGOFMM.PyDistData_RIDS(comm, problem_size, 1, iset=rids.astype('int32'), darr=local_ones)

    #generate test vector of half ones and half zeros
    rank = comm.Get_rank()
    if rank==0:
        local_test1 = np.ones([local_rows, 1], dtype='float32', order='F')
    else:
        local_test1 = np.zeros([local_rows, 1], dtype='float32', order='F')
    global_test1 = PyGOFMM.PyDistData_RIDS(comm, problem_size, 1, iset=rids.astype('int32'), darr=local_test1)


    rank = comm.Get_rank()
    if rank==1:
        local_test2 = np.ones([local_rows, 1], dtype='float32', order='F')
    else:
        local_test2 = np.zeros([local_rows, 1], dtype='float32', order='F')
    global_test2 = PyGOFMM.PyDistData_RIDS(comm, problem_size, 1, iset=rids.astype('int32'), darr=local_test2)
    gids = gofmm.getGIDS()
    

    DT1 = gofmm.mult_hmlp(global_test1)
    DT2 = gofmm.mult_hmlp(global_test2)
    D = gofmm.mult_hmlp(Ones)
    D = gofmm.mult_hmlp(Ones)

    KH = gofmm.mult_hmlp(H)
    #print("RID: ", rids[0])
    if True:
        np.savetxt('D_'+str(rank), D.toArray())
        np.savetxt('rids_'+str(rank), rids)
    if 139 in rids:
        print("RID: 139 on rank", rank)
        print("\tOnes on Rank0, Zeros elsewhere")
        print("\tResult of K times [111, 000]: ", DT1[139, 0])
        print("\tZeros on Rank0, Ones elsewhere")
        print("\tResult of K times [000, 111]: ", DT2[139, 0])
        print("\tArray of all ones")
        print("\tResult of K times [1111,1111]: ", D[139, 0])    


    if 278 in rids:
        print("RID: 278 on rank", rank)
        print("\tOnes on Rank0, Zeros elsewhere")
        print("\tResult of K times [111, 000]: ", DT1[278, 0])
        print("\tZeros on Rank0, Ones elsewhere")
        print("\tResult of K times [000, 111]: ", DT2[278, 0])
        print("\tArray of all ones")
        print("\tResult of K times [1111,1111]: ", D[278, 0])
     
    if 556 in rids:
        print("RID: 556 on rank", rank)
        print("\tOnes on Rank0, Zeros elsewhere")
        print("\tResult of K times [111, 000]: ", DT1[556, 0])
        print("\tZeros on Rank0, Ones elsewhere")
        print("\tResult of K times [000, 111]: ", DT2[556, 0])
        print("\tArray of all ones")
        print("\tResult of K times [1111,1111]: ", D[556, 0])
    
     
    D_RBLK = PyGOFMM.PyDistData_RBLK(comm, problem_size, 1)
    D_RBLK.loadRIDS(D)

    ridcheck = 139
    if rank == ridcheck % size:
        print("RBLK check. Rank: ", rank, "RID: ", ridcheck)
        print("Result all ones: ", D_RBLK[139, 0])

    localsum = 0
    for j in gids:
        localsum += gofmm.K.getValue(ridcheck, j) * 1
    globalsum = np.asarray(0).astype('float32')
    localsum = np.asarray(localsum).astype('float32')
    comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    if rank == ridcheck % size:
        print("Direct Multiplication Value: ", globalsum)

    #Generate HKH, HDH, DKH
    HKH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    HDH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    DKH = np.zeros([local_rows, nclasses], dtype='float32', order='F')

    #TODO: Recheck if the indicies are correct
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
    return (D, DKH, HKH, HDH) 

petsc4py.init(comm=MPI.COMM_WORLD)
nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

N_per = (int)(4000/nprocs)
N = N_per*nprocs
d = 3
print("Points per processor", N_per)
print("Total problem size", N)
print("I'm rank: ", rank)
conf = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.0001, 0.01, True)
comm_petsc = PETSc.COMM_WORLD
comm_mpi = MPI.COMM_WORLD
np.random.seed(10)
#Set up artificial points (two classes)
#class_1 = np.random.randn(d, (int)(np.floor(N_per/2)))+2
#class_2 = np.random.randn(d, (int)(np.ceil(N_per/2)))
#test_points = np.concatenate((class_1, class_2), axis=1) #data points shape = (d, N_per)


#Set up artificial points (two classes) that are the same no matter how many processors
class_1 = np.random.randn(d, (int)(np.floor(N/2)))+5
class_2 = np.random.randn(d, (int)(np.ceil(N/2)))
test_points = np.concatenate((class_1, class_2), axis=1) #data points shape = (d, N_per)

print("This is the point with GID 139 at creation: ", test_points[:, 139])
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
#print(test_points)
source_points = PETSc.Vec().createWithArray(test_points)
true_classes = PETSc.Vec().createWithArray(true_classes)
classes = PETSc.Vec().createWithArray(classes)


gofmm = GOFMM_Kernel(comm_mpi, N, d, source_points, config=conf) #set up python context for gofmm
redistributed_points = gofmm.redistribute(source_points, d, N, nper=N_per, form='hmlp') #redistribute to HMLP Dist Data 
redistributed_true_classes = gofmm.redistribute(true_classes, 1, N, nper=N_per, form='hmlp')
redistributed_classes = gofmm.redistribute(classes, 1, N, nper=N_per, form="hmlp")

#check that classes are still with corresponding points
points_rids = redistributed_points.getRIDS()
classes_rids = redistributed_points.getRIDS()
#print(np.array_equal(points_rids, classes_rids))

#for i in points_rids:
print("This is the point after redistribution")
if 139 in points_rids:
    print("RID: ", 139, " Class: ", redistributed_true_classes[139, 0], "Point: (", redistributed_points[139, 0], ", ", redistributed_points[139, 1], ", ", redistributed_points[139, 2], ") is on rank", rank)

#set up gofmm operator
A = PETSc.Mat().createPython( [N, N], comm=comm_petsc)
A.setPythonContext(gofmm)
A.setUp()

#Renaming for convenience
classes = redistributed_classes
points = redistributed_points
true_classes = redistributed_true_classes

#####################
#Start K Means

#Let 
#   - DKH = D^-1 K H
#   - HKH = H^t K H
#   - HDH = H^t D H
k = 2

#Main loop

#print(classes.toArray())
#print(true_classes.toArray())
#print(np.array_equal(classes, true_classes))
#print(classes.getRIDS())
np_points = points.toArray()
np_classes = classes.toArray()
#plt.scatter(np_points[:, 0], np_points[:, 1], c=np_classes.flatten())
#plt.show()

maxitr = 1
rids = classes.getRIDS()
local_rows = len(rids)
problem_size = N

local_ones = np.ones([local_rows, 1], dtype='float32', order='F')
Ones = PyGOFMM.PyDistData_RIDS(comm_mpi, problem_size, 1, iset=rids.astype('int32'), darr=local_ones)
D = gofmm.mult_hmlp(Ones)
post_rids, pre_rids = learnMapping(comm_mpi, D)
print("HEY LOOK HERE")
print(post_rids)
print(pre_rids)

for i in range(maxitr):
    #Generate these matricies
    (D, DKH, HKH, HDH) = KMeansPrep(A, classes, k)

    #Get diagonal (alternatively just assume Diag=1)
    #Diag = A.getDiagonal()
    rids = points.getRIDS()
    local_rows = len(rids)
    d = points.cols()
    Diag = np.ones([local_rows, 1])
    #Compute similarity 
    Similarity = np.zeros([local_rows, k], dtype='float32', order='F')
    for i in range(local_rows):
        for p in range(k):
            Similarity[i, p] = Diag[i]/(D[rids[i], 0]*D[rids[i], 0]) - 2 * DKH[i, p]/HDH[p, p] + HKH[p, p]/(HDH[p, p] * HDH[p, p])

    #print(Similarity)
    #Compute fit Need complement of H_local
    #quality = np.zeros([1, k], dtype='float32', order='F')
    #for i in range(k):
    #    quality[1, i] += HKH[i, i]

    for i in range(local_rows):
        classes[rids[i], 0] = np.argmin(Similarity[i, :])+1
    #print(classes.toArray())
    #print(classes[rids[0], 0])
    #print(classes[rids[1], 0])

np_classes = classes.toArray()
np_true_classes = true_classes.toArray()

#plt.scatter(np_points[:, 0], np_points[:, 1], c=np_classes.flatten())
#plt.show()


#print(classes.toArray())
#print(true_classes.toArray())
#print(np.array_equal(classes, true_classes))
