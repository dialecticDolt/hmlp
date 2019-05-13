#Import from PyGOFMM
cimport pygofmm.core as PyGOFMM
import pygofmm.core as PyGOFMM #TODO: Why can't I see cdef declarations from core.pxd ?????

from pygofmm.DistData cimport *
from pygofmm.CustomKernel cimport *

#Import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
from mpi4py import MPI

#Import from cython: cpp
from libcpp.pair cimport pair
from libcpp cimport vector
from libcpp.map cimport map

#Import from cython: c
from libc.math cimport sqrt
from libc.string cimport strcmp
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.stdio cimport printf
from libcpp cimport bool as bool_t

#Import from cython
import cython
from cython.operator cimport dereference as deref
from cython.view cimport array as cvarray
from cython.parallel cimport prange

#Numpy
import numpy as np
cimport numpy as np
np.import_array()

#PETSc and SLEPc
import petsc4py
from slepc4py import SLEPc
from petsc4py import PETSc

def testFunction(PyGOFMM.PyRuntime testparam):
    testparam.init()
    testparam.finalize()

#NOTE: Hey, check this out, apparently our toArray() isn't making a copy
#      using b afterwards is causing a segfault for come reason
def testDistDataArray(PyGOFMM.PyDistData_RIDS b):
    cdef float[:, :] t = b.toArray()
    print(t[0, 0])
    t[0, 0] = 1000
    print(t[0, 0])
    t = b.toArray()
    print(t[0, 0])
    print("Done")
#(Pure Python) KMeans
#A lot of this is inefficient and can be fixed once cdefs are visible from core.pxd

#Assume input is in the same cyclic ordering as the columns of K
#This ordering is given by the gids parameter
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def KMeans(PyGOFMM.KernelMatrix K, int nclasses, gids, classvec=None, maxiter=10):
    cdef int N, n_local
    cdef int c, i, j, k, p, itr
    cdef MPI.Comm comm
    cdef int[:] rids = K.getTree().getGIDS().astype('int32') #get local row gofmm-tree ordering
    
    N = K.getSize()
    n_local = len(rids)
    comm = K.getComm()

    #initialize the classvector randomly if not given
    if classvec is None:
        classvec = np.zeros([n_local], order='F', dtype='float32')
        for i in xrange(n_local):
            classvec[i] = np.random.randint(1, nclasses+1)
        GOFMM_classes = PyGOFMM.PyDistData_RIDS(comm, N, 1, iset=rids, arr=classvec)
    else:
        #load class data into PyGOFMM DistData object, NOTE: two copies are made here
        Temp = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, arr=classvec, iset=gids)
        GOFMM_classes = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, iset=rids)
        GOFMM_classes.redistribute(Temp)

    #initialize class indicator block
    cdef float[:, :] H_local = np.zeros([n_local, nclasses], dtype='float32', order='F')
    for i in xrange(n_local):
        H_local[i, <int>(GOFMM_classes[rids[i], 0] - 1)] = 1.0

    #copy class indicator block to DistData object
    H = PyGOFMM.PyDistData_RIDS(comm, N, nclasses, iset=rids, darr=H_local)

    #form D
    print(N)
    print(n_local)
    print(len(rids))
    cdef float[:, :] local_ones = np.ones([n_local, 1], dtype='float32', order='F')
    Ones = PyGOFMM.PyDistData_RIDS(comm, N, 1, iset=rids, darr=local_ones)
    D = K.evaluate(Ones)

    cdef float[:, :] npD = D.toArray() #create local numpy copy in order to use shared memory parallelism for similarity computation
    cdef float[:, :] Diag = np.ones([n_local, 1], dtype='float32', order='F')
    
    #allocate storage for lookup matricies
    cdef float[:, :] HKH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] HKH = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] HDH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] HDH = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] DKH = np.zeros([n_local, nclasses], dtype='float32', order='F')
    cdef float[:, :] Similarity = np.zeros([n_local, nclasses], dtype='float32', order='F')

    #start main loop
    for itr in xrange(maxiter):
        
        #update DKH, HKH, HDH
        KH = K.evaluate(H)
        
        #TODO: Replace this with shared memory parallel version or MKL
        for i in xrange(nclasses):
            for j in xrange(nclasses):
                for r in rids:
                    HKH_local[i, j] += H[r, j]*KH[r, i]
                    HDH_local[i, j] += H[r, j]*D[r, 0]*H[r, i]

        #TODO: Replace this with shared memory parallel version or MKL
        for i in xrange(n_local):
            for j in xrange(nclasses):
                DKH[i, j] = 1/npD[i, 0] * KH[rids[i], j]

        HKH = np.zeros([nclasses, nclasses], dtype='float32', order='F')
        HDH = np.zeros([nclasses, nclasses], dtype='float32', order='F')
        
        comm.Allreduce(HKH_local, HKH, op=MPI.SUM)
        comm.Allreduce(HDH_local, HDH, op=MPI.SUM)
        
         #update similarity
        for i in prange(n_local, nogil=True):
            for p in xrange(nclasses):
                Similarity[i, p] = Diag[i, 0]/(npD[i, 0]*npD[i, 0]) - 2*DKH[i, p]/HDH[p, p] + HKH[p, p]/(HDH[p, p]* HDH[p, p])
        
        print(np.asarray(Similarity))
        #update classvector 
        for i in xrange(n_local):
            GOFMM_classes[rids[i], 0] = np.argmin(Similarity[i, :])+1
        print(GOFMM_classes.toArray())
        #update class indicator matrix H
        H_local = np.zeros([n_local, nclasses], dtype='float32', order='F')
        for i in xrange(n_local):
            H_local[i, <int>(GOFMM_classes[rids[i], 0] - 1)] = 1.0
        
        print(np.asarray(H_local))
        #copy class indicator matrix to DistData object
        H = PyGOFMM.PyDistData_RIDS(comm, N, nclasses, iset=rids, darr=H_local)

    classes = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, iset=gids)
    classes.redistribute(GOFMM_classes)

    return classes.toArray()

#pure python implementation of KDE
def KDE(PyGOFMM.KernelMatrix K, int nclasses,int[:] gids, float[:] classvec):
    cdef int N, n_local
    cdef int i,ci,j
    cdef MPI.Comm comm
    cdef int[:] rids = K.getTree().getGIDS().astype('int32') #get local row gofmm-tree ordering
    
    N = K.getSize()
    n_local = len(rids)
    comm = K.getComm()

    #load class data into PyGOFMM DistData object, TODO: necessary for creation of ww_hmlp?
    classes_user = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, arr=classvec, iset=gids)
    classes_hmlp = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, iset=rids)
    classes_hmlp.redistribute(classes_user)

    #initialize class indicator block
    cdef float[:, :] ww_hmlp_loc= np.zeros([n_local, nclasses], dtype='float32', order='F')
    for i in xrange(n_local): #TODO fix loop?
        ww_hmlp_loc[i, <int>(classes_hmlp[rids[i], 0] - 1)] = 1.0

    #copy class indicator block to DistData object
    ww_hmlp = PyGOFMM.PyDistData_RIDS(comm, N, nclasses, iset=rids, darr=ww_hmlp_loc)

    # Compute multiply
    density_hmlp = K.evaluate(ww_hmlp)
    density_user = PyGOFMM.PyDistData_RIDS(comm,m = N,n=nclasses, iset = gids)
    density_user.redistribute(density_hmlp)

    # output density after redistributing and subtracting local contributions
    return density_user.toArray() - ww_hmlp_loc #TODO current default removes self interactions. keep?

def TestKDE(PyGOFMM.KernelMatrix K, int nclasses, int [:] gids, float[:] classvec, float[:,:] Xte):
    cdef int N, n_local
    cdef int i,ci,j
    cdef MPI.Comm comm
    cdef int[:] rids = K.getTree().getGIDS().astype('int32') #get local row gofmm-tree ordering
    
    N = K.getSize()
    n_local = len(rids)
    comm = K.getComm()

    #load class data into PyGOFMM DistData object, TODO: necessary for creation of ww_hmlp?
    classes_user = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, arr=classvec, iset=gids)
    classes_hmlp = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, iset=rids)
    classes_hmlp.redistribute(classes_user)

    #initialize class indicator block
    cdef float[:, :] ww_hmlp_loc= np.zeros([n_local, nclasses], dtype='float32', order='F')
    for i in xrange(n_local): #TODO fix loop?
        ww_hmlp_loc[i, <int>(classes_hmlp[rids[i], 0] - 1)] = 1.0

    #copy class indicator block to DistData object
    ww_hmlp = PyGOFMM.PyDistData_RIDS(comm, N, nclasses, iset=rids, darr=ww_hmlp_loc)

    # Compute multiply
    cdef PyGOFMM.PyData Xte_py = PyGOFMM.PyData( m = Xte.shape[0], n = Xte.shape[1], darr = Xte)
    density_test = K.evaluateTest(Xte_py, ww_hmlp)

    # return
    return density_test.toArray()

cdef class GOFMM_Handler(object):
    cdef PyGOFMM.KernelMatrix K
    cdef MPI.Comm comm
    cdef int size
    cdef PyGOFMM.PyDistData_RIDS D
    cdef int[:] rids
    cdef bool_t norm

    def __init__(self, K, norm=False):
        self.K = K
        self.size = K.getSize()
        self.comm = K.getComm()
        self.rids = K.getTree().getGIDS().astype('int32')
        n_local = len(self.rids)
        self.norm = norm
        if self.norm:
            local_ones = np.ones(n_local, dtype='float32', order='F')
            ones = PyGOFMM.PyDistData_RIDS(self.comm, n_local, 1, arr=local_ones, iset=self.rids)
            self.D = self.K.evaluate(ones)
    
    def getSize(self):
        return self.size

    def getMPIComm(self):
        return self.comm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def mult(self, mat, X, Y):
        with X as x:
            with Y as y:
                n = len(x)
                nrhs = 1
                GOFMM_x = PyGOFMM.PyDistData_RIDS(self.comm, m=self.size, n=nrhs, tree=self.K.getTree(), arr=x.astype('float32'))
                GOFMM_y = self.K.evaluate(GOFMM_x)
                for i in range(len(y)):
                    if self.norm:
                        y[i] = 1/self.D[self.rids[i], 0] * GOFMM_y[self.rids[i], 0]
                    else:
                        y[i] = GOFMM_y[self.rids[i], 0]
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
        cdef int n, nrhs
        with X as x:
            with B as b:
                n = len(b)
                nrhs = 1
                GOFMM_b = PyGOFMM.PyData(m=n, n=nrhs, arr=x.astype('float32'))
                self.K.solve(GOFMM_b, 1)
                for i in range(len(x)):
                    x[i] = GOFMM_b[i, 0]

cdef class GOFMM_Kernel(object):

    def __init__(self, comm, N, d, sources, targets=None, kstring="GAUSSIAN", config=None, petsc=True, bwidth=1.0):
        self.rt = PyGOFMM.PyRuntime()
        self.rt.init_with_MPI(comm)
        self.comm = comm
        if not petsc:
            self.K = PyGOFMM.KernelMatrix(comm, sources, targets, kstring, config, bwidth)
            self.size = N
        if petsc:
            if(targets==None):
                with sources as src:
                    GOFMM_src = PyGOFMM.PyDistData_CBLK(comm, d, N, arr=src.astype('float32'))
                    self.K = PyGOFMM.Kernelmatrix(comm, GOFMM_src, targets, kstring, config, bwidth)
            else:
                with sources as src:
                    with targets as trg:
                        GOFMM_src = PyGOFMM.PyDistData_CBLK(comm, d, N, arr=src.astype('float32'))
                        GOFMM_trg = PyGOFMM.PyDistData_CBLK(comm, d, N, arr=trg.astype('float32'))
                        self.K = PyGOFMM.KernelMatrix(comm, GOFMM_src, GOFMM_trg, kstring, config, bwidth)
        self.K.compress()

    def __dealloc__(self):
        self.rt.finalize()

    def getSize(self):
        return self.size

    def getMPIComm(self):
        return self.comm

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
    

def SpecCluster(PyGOFMM.KernelMatrix K, int nclasses, int[:] gids):
        
    cdef int N, n_local
    cdef int c, i, j, k, p, itr
    cdef MPI.Comm comm
    cdef int[:] rids = K.getTree().getGIDS().astype('int32') #get local row gofmm-tree ordering
    
    N = K.getSize()
    n_local = len(rids)
    comm = K.getComm()

    gofmm = GOFMM_Handler(K, norm=True)
    petsc4py.init(comm=comm)
    comm_petsc = MPI.COMM_WORLD
    A = PETSc.Mat().createPython([N, N], comm = comm_petsc)
    A.setPythonContext(gofmm)
    A.setUp()
    x, b = A.createVecs()
    x.set(1.0)

    E = SLEPc.EPS()
    E.create()
    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    E.setFromOptions()
    E.setDimensions(nclasses)
    E.setDeflationSpace(x)
    E.solve()
    Print = PETSc.Sys.Print

    its = E.getIterationNumber()
    eps_type = E.getType()
    nev, ncv, mpd = E.getDimensions()
    tol, maxit = E.getTolerances()
    nconv = E.getConverged()
    print(nconv)
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()
    for i in range(nconv):
        res = E.getEigenpair(i, vr, vi)
        print(i)
        print("--------")
        print(res.real)
        print(res.imag)
