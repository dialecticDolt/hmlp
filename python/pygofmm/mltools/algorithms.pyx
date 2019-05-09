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
from libc.math cimport sqrt, log2
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

#flush
import sys


def testFunction(PyGOFMM.PyRuntime testparam):
    testparam.init()
    testparam.finalize()

cdef getCBLKOwnership(int N, int rank, int nprocs):
    cblk_idx = np.asarray(np.arange(rank, N, nprocs), dtype='int32', order='F')
    return cblk_idx

def CBLK_Distribute(MPI.Comm comm, float[:, :] points):
    cdef int N, d, nprocs, rank
    N = np.size(points[:, 0])
    d = np.size(points[0, :])
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    CBLK_points = points[:, getCBLKOwnership(N, rank, nprocs)]
    CBLK_points = np.asarray(CBLK_points, dtype='float32', order='F')
    sources = PyGOFMM.PyDistData_CBLK(comm, d, N, darr=CBLK_points)
    return sources

#def GOFMM_Distribute(MPI.Comm comm, float[:, :] points):



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



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef computeClusteringCost(MPI.Comm comm, float[:, :] points, int nclasses, float [:, :] centers):
    cdef float minimumDist, tempDist, local_sum, global_sum 
    cdef int n_local, d, r, i, j
    global_sum = 0.0
    n_local = np.shape(points)[1]
    d = np.shape(points)[0] 
    for r in prange(n_local, nogil=True):
        minimumDist = -1.0
        for i in xrange(nclasses):
            tempDist = 0.0
            for j in xrange(d):
                #TODO: Change this so its column major efficient?
                tempDist = tempDist + (points[r, j] - centers[i, j])* (points[r, j] - centers[i, j])
            if (tempDist < minimumDist) or (minimumDist == -1):
                minimumDist = tempDist
        local_sum += minimumDist

    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    return global_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float minDist(int d, float[:] point, int nclasses, float[:, :] centers) nogil:
    cdef int i, j
    cdef float minimumDist, tempDist
    minimumDist = -1
    for i in xrange(nclasses):
        tempDist = 0.0
        for j in xrange(d):
            tempDist = tempDist +  (point[j] - centers[j, i])*(point[j] - centers[j, i])
        if (tempDist < minimumDist) or (minimumDist == -1):
            minimumDist = tempDist
    return minimumDist


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def local_bsearch(float target, float[:] arr):
    cdef int n_local, mid, start, end
    n_local = len(arr)
    if n_local < 2:
        return 0
    start = 1
    end = n_local - 1
    while ( start <= end ):
        mid = (start + end)/2
        if (arr[mid] > target):
            end = mid-1
        elif (arr[mid] < target):
            start = mid + 1
        else:
            return mid
    if (mid>0) & (arr[mid-1] < target < arr[mid]):
        return mid-1
    else:
        return mid

#TODO: Make this inplace for n_local not a power of 2
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def local_prefix_sum(float[:] a):
    cdef int i, j, d, writestride, readstride, depth, Npack
    cdef int n_local = len(a)
    cdef float[:] arr
    Npack = n_local
    if (1<<(<int>np.log2(n_local)) != n_local):
        Npack = 1<<(<int>(np.log2(n_local))+1)
    arr = np.zeros([Npack], dtype='float32', order='F')
    
    for i in prange(n_local, nogil=True):
        arr[i] = a[i]

    writestride = 0
    readstride = 0
    depth = <int>np.log2(Npack)
    cdef float t
    
    #Downward pass
    for d in xrange(depth):
        writestride = 1<<(d+1)
        readstride = 1<<d
        for j in prange(0, n_local, writestride, nogil=True):
            arr[j+writestride-1] = arr[j+writestride-1] + arr[j+writestride-1 - readstride]
    
    arr[Npack-1] = 0 
    
    #Upward pass
    for d in xrange(depth-1, -1, -1):
        writestride = 1<<(d+1)
        readstride = 1<<d
        for j in prange(0, n_local, writestride, nogil=True):
            t = arr[j+writestride-1-readstride]
            arr[j+writestride-1-readstride] = arr[j+writestride-1]
            arr[j+writestride-1] = t + arr[j+writestride-1]
    return arr[0:n_local]

#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
#cdef prefix_sum(comm, float[:] arr):

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def local_kmeans_pp(float[:, :] points, int nclasses):
    cdef int n_local, d, indx, i, r
    cdef float cost, dist, u
    n_local = np.shape(points)[1]
    d = np.shape(points)[0]

    cdef float[:, :] centerlist = np.zeros([d, nclasses], dtype='float32', order='F')
    cdef float[:] dlist = np.zeros([n_local], dtype='float32', order='F')
    #start with a random point
    r = np.random.randint(1, n_local)
    centerlist[:, 0] = points[:, r]
    for i in xrange(1, nclasses):

        for r in prange(n_local, nogil=True):
            dlist[r] = minDist(d, points[:, r], i, centerlist)
        cost = np.sum(dlist)
        dlist = local_prefix_sum(dlist)
        for r in prange(n_local, nogil=True):
            dlist[r] = dlist[r]/cost;
        u = np.random.rand()
        indx = local_bsearch(u, dlist)
        centerlist[:, i] = points[:, indx]
    
    return np.asarray(centerlist, dtype='float32', order='F')


cdef shareNumber(MPI.Comm comm, u, int r):
    a = np.array(u)
    comm.Bcast(a, root=r)
    return a


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def distributed_kmeans_pp(MPI.Comm comm, float[:, :] points, int nclasses):
    comm.Barrier()
    cdef int N, n_local, d, indx, i, r, k
    cdef int size, rank
    cdef float u
    size = comm.size
    rank = comm.rank

    n_local = np.shape(points)[1]
    d = np.shape(points)[0]
   
    #print(rank, ": Starting distributed k means++")
    #sys.stdout.flush() 
    cdef float[:, :] centerlist = np.zeros([d, nclasses], dtype='float32')

    #collect information to uniformly sample first point
    N = 0
    sendbuf = np.zeros([1], dtype='int32') + n_local
    recvbuf = np.zeros([size], dtype='int32')
    recvbuf = comm.allgather(sendbuf)
    #print(rank, ": finished sharing n_local information")

    #sys.stdout.flush() 
    local_lengths = np.concatenate(recvbuf)
    N = np.sum(local_lengths)
    
    local_prob = np.asarray(np.asarray(local_lengths, dtype='float32')/N, dtype='float32')
    
    if rank == 0:
        u = np.random.rand()
    u = shareNumber(comm, u, 0)
    indx = local_bsearch(u, local_prob)
    
    #print(rank, ": Selecting random start")
    #print(rank, u)
    #sys.stdout.flush() 
    #start with a random point as the center
    if rank == indx:
        i = np.random.randint(0, n_local)
        centerlist[:, 0] = points[:, i]
    
    comm.Barrier()
    comm.Bcast(centerlist, root=indx)
    
    #print(rank, ": Finished broadcasting random start")

    #sys.stdout.flush() 
    #print(np.asarray(centerlist))
    
    cdef float[:] dlist = np.zeros([n_local], dtype='float32')
    cdef float[:] plist = np.zeros([size], dtype='float32')
    for k in xrange(1, nclasses):
        #compute local dlist
        for r in prange(n_local, nogil=True):
            dlist[r] = minDist(d, points[:, r], k, centerlist)
    
        #print(rank, ": finished computing min distances")
        #sys.stdout.flush() 

        #compute cost
        local_cost = np.array(np.sum(dlist), dtype='float32')
        cost = np.array(0.0, dtype='float32')
        comm.Barrier()
        comm.Allreduce(local_cost, cost, op=MPI.SUM)
        #print(rank, ": finished all reduce on local_cost")
        #sys.stdout.flush() 

        #compute plist (to determine which rank to sample from)
        sendbuf = np.zeros([1], dtype='float32') + local_cost
        recvbuf = np.zeros([size], dtype='float32')
        recvbuf = comm.allgather(sendbuf)
        #print(rank, ": finised sharing local_cost")
        #sys.stdout.flush() 

        plist = np.concatenate(recvbuf)
        plist = local_prefix_sum(plist)
        plist = np.asarray(plist)/cost
        
        #sample based on local cost per rank
        if rank==0:
            u = np.random.rand()
        u = shareNumber(comm, u, 0)
        indx_g = local_bsearch(u, plist)
        #print(rank, indx_g)
        #sys.stdout.flush()
        if rank == indx_g:
            #print("rank", rank)
            dlist = local_prefix_sum(dlist)
            dlist = np.asarray(dlist)/local_cost
            indx = local_bsearch(u, dlist)
            #print(rank, indx)
            #print(rank, np.shape(dlist))
            #print(rank, np.asarray(points[:, indx]))
            centerlist[:, k] = points[:, indx]
            #print(rank, np.asarray(points[:, indx]))
            #print(rank, np.asarray(centerlist))

        #print(rank, np.asarray(centerlist))
        #print(rank, ": finished picking point") 
        #sys.stdout.flush() 
        #print(rank, indx_g)
        comm.Barrier()
        comm.Bcast(centerlist, root=indx_g)
        #print(rank, ": finished Bcast")
        
        #sys.stdout.flush() 
        #print(rank, np.asarray(centerlist))
        #print(rank, k)
    #return np.asarray(centerlist, order='F')
    return centerlist

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def KMeans(MPI.Comm comm, float[:, :] points, int nclasses, classvec = None, maxiter=10, init="random"):
    cdef int n_local
    cdef int i,l, d, index
    cdef int nprocs, rank

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    n_local = np.shape(points)[1]
    d = np.shape(points)[0] 


    if classvec is None:
        classvec = np.zeros([n_local], order='F', dtype='int32')
        #for i in xrange(n_local):
        #    classvec[i] = np.random.randint(1, nclasses+1)
    else:
        classvec = np.asarray(classvec, dtype='int32', order='F')

    cdef int[:] classes = classvec
    cdef float[:, :] centroids = np.zeros([d, nclasses], dtype='float32')
    cdef float[:, :] temp_centroids = np.zeros([d, nclasses], dtype='float32')

    if init == "random":
        for i in range(nclasses):
            if rank ==0:
                u = np.random.randint(0, nprocs)
            u = shareNumber(comm, u, 0)
            if rank == u:
                index = np.random.randint(0, n_local)
                temp_centroids[:, i] = points[:, index]
        comm.Allreduce(temp_centroids, centroids, op=MPI.SUM)

    elif init == "++":
        centroids = distributed_kmeans_pp(comm, points, nclasses)

    cdef float distance, temp_dist
    for l in xrange(maxiter):
        #reassign classes
        updateClasses(points, classes, centroids, nclasses)
        #find centroids
        centroids = computeCenters(comm, points, classes, nclasses)
    return np.asarray(classes)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef updateClasses(float[:, :] points, int[:] classes, float[:, :] centroids, int nclasses=2):
    cdef int r, k, i, c, d , n_local

    n_local = np.shape(points)[1]
    d = np.shape(points)[0] 
    cdef float temp_dist, distance
    #for each point compute L2 distance to each centroid
    #r = gid, k = class, i=dim
    for r in prange(n_local, nogil=True):
        distance = -1
        for k in xrange(nclasses):
            temp_dist = 0.0
            for i in xrange(d):
                #print(centroids[i, k])
                #sys.stdout.flush()
                temp_dist = temp_dist + (points[i, r] - centroids[i, k])* (points[i, r] - centroids[i, k])
            if (temp_dist < distance) or (distance == -1):
                distance = temp_dist
                c = k
        classes[r] = c + 1 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef computeCenters(MPI.Comm comm, float[:, :] points, int[:] classes, int nclasses=2):
    cdef int N, n_local, nprocs
    cdef int c, r, l, d, i
    
    d = np.shape(points)[0]
    
    n_local = len(classes)
    nprocs = comm.Get_size()
    cdef float[:, :] centroids_local = np.zeros([d, nclasses], dtype='float32')
    cdef float[:, :] centroids = np.copy(centroids_local)
    cdef float[:] counts = np.zeros([nclasses], dtype='float32')

    for r in prange(n_local, nogil=True):
        c = classes[r]-1
        #print(np.asarray(points[:, r]))
        #print("--")
        for l in xrange(d):
            centroids_local[l, c] += points[l, r]
        counts[c] += 1
   
    for i in xrange(nclasses):
        for l in prange(d, nogil=True):
            if counts[i]==0:
                centroids_local[l, i] = 0
                continue
            centroids_local[l, i] = centroids_local[l, i]/(counts[i]*nprocs)

    comm.Allreduce(centroids_local, centroids, op=MPI.SUM)
    return centroids

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def NMI(MPI.Comm comm, int[:] truth, int[:] clusters, int nclasses):
    #Compute entropy of true labelings
    cdef int i, j, k, c
    cdef float t, h_y, h_c
    h_y = 0.0
    h_c = 0.0
    cdef int n_local = len(truth)
    N_total = np.array(0)
    local_len = np.array(n_local)
    comm.Allreduce(local_len, N_total, op=MPI.SUM)
    cdef float[:] truth_counts_local = np.zeros([nclasses], dtype='float32')
    cdef float[:] truth_counts = np.copy(truth_counts_local)
    for i in xrange(n_local):
        c = truth[i]-1
        truth_counts_local[c] += 1
    
    comm.Allreduce(truth_counts_local, truth_counts, op=MPI.SUM)
    for j in xrange(nclasses):
        t = (truth_counts[j])/N_total
        if t>0:
            h_y -= t * log2(t)
        
    cdef float[:] clust_counts_local = np.zeros([nclasses], dtype='float32')
    cdef float[:] clust_counts = np.copy(truth_counts_local)
    for i in xrange(n_local):
        c = clusters[i]-1
        clust_counts_local[c] += 1

    comm.Allreduce(clust_counts_local, clust_counts, op=MPI.SUM)
    for j in xrange(nclasses):
        t = clust_counts[j]/N_total
        if t>0:
            h_c -= t * log2(t)

    cdef float[:, :] counts_local = np.zeros([nclasses, nclasses], dtype='float32')
    cdef float[:, :] counts = np.copy(counts_local)
    for i in xrange(n_local):
        c = clusters[i]-1
        k = truth[i]-1
        counts_local[c, k] += 1

    print(np.asarray(counts_local))
    comm.Allreduce(counts_local, counts, op=MPI.SUM)
    cdef float[:] I = np.zeros([nclasses], dtype='float32')
    for i in xrange(nclasses):
        for j in xrange(nclasses):
            t = counts[i, j]/clust_counts[i]
            if t>0:
                I[i] += t*log2(t)
    cdef float h_yc = 0.0
    for i in xrange(nclasses):
        h_yc -= clust_counts[i]/N_total * I[i]

    return (2*(h_y - h_yc)/(h_y + h_c))



#(Pure Python) KMeans
#A lot of this is inefficient and can be fixed once cdefs are visible from core.pxd
#Assume input is in the same cyclic ordering as the columns of K
#This ordering is given by the gids parameter
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def KKMeans(PyGOFMM.KernelMatrix K, float[:, :] points, int nclasses, gids, classvec=None, maxiter=10, init="random"):
    cdef int N, n_local, d
    cdef int c, i, j, k, p, itr, u
    cdef MPI.Comm comm
    cdef int[:] rids = K.getTree().getGIDS().astype('int32') #get local row gofmm-tree ordering
    
    N = K.getSize()
    n_local = len(rids)
    comm = K.getComm()
    nprocs = comm.size
    rank = comm.rank
    d = len(points[:, 0])
   
    cdef float[:, :] centers = np.zeros([d, nclasses], dtype='float32')
    cdef int[:] center_ind = np.zeros([nclasses], dtype='int32')
    cdef float[:] dlist = np.zeros([n_local], dtype='float32')
    cdef float minimumDist, currentDist
    init = "random_points"
    if classvec is None:
        classvec = np.zeros([n_local], order='F', dtype='float32')
        for i in xrange(n_local):
            classvec[i] = np.random.randint(1, nclasses+1)
        GOFMM_classes = PyGOFMM.PyDistData_RIDS(comm, N, 1, iset=rids, arr=classvec)
    if classvec is None and init=="random_points":
        centers = np.zeros([nclasses])
        if rank == 0:
            centers = np.random.randint(0, n_local, size=nclasses)
        comm.Bcast(classvec, root=0)
        classvec = np.zeros([n_local], dtype='float32')
        for j in xrange(n_local):
            mD = -1
            for k in xrange(nclasses):
                cind = centers[k] 
                cD = 2 - 2*K[cind, j]
                if (cD < mD) or (mD==-1):
                    mD = cD
                    c = k
            classvec[j] = c
        GOFMM_classes = PyGOFMM.PyistData_RIDS(comm, N, 1, iset=rids, arr=classvec)

    #    classvec = np.zeros([n_local], dtype='float32')
    #    if rank ==0:
    #        for k in xrange(nclasses):
    #            center_ind[k] = np.random.randint(0, n_local)
    #    comm.Bcast(center_ind, root=0)
        #Pick k random points to be the centers
        #for k in xrange(nclasses):
        #    if rank == 0:
        #        u = np.random.randint(0, nprocs)
        #    u = shareNumber(comm, u, 0)
        #    if u==rank:
        #        u = np.random.randint(0, n_local)
        #        centers[:, k] = points[:, u]

    #Initialize classvector from centers 
    #    for j in xrange(n_local):
    #        minimumDist = -1.0
    #        for k in xrange(nclasses):
    #            c_ind = center_ind[k]
    #            currentDist = K[c_ind, c_ind] + K[j, j] - 2*K[c_ind, j]
    #            if (currentDist < minimumDist) or (minimumDist == -1.0):
    #                minimumDist = currentDist
    #                c = k
    #        classvec[j] = c
    #    print(np.asarray(classvec))
    #    sys.stdout.flush()
    #    print(len(classvec))
    #    print(len(rids))
    #    GOFMM_classes = PyGOFMM.PyDistData_RIDS(comm, N, 1, iset=rids, arr=classvec)
    #    print("hey")
    if classvec is not None:
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
        
        for i in xrange(nclasses):
            for j in xrange(nclasses):
                HKH_local[i, j] = 0.0
                HDH_local[i, j] = 0.0
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
                #Ignore the degenerate case (probably not the best solution)
                if (npD[i, 0] == 0) or (HDH[p, p]==0):
                    Similarity[i, p] = 0
                    continue
                Similarity[i, p] = Diag[i, 0]/(npD[i, 0]*npD[i, 0]) - 2*DKH[i, p]/HDH[p, p] + HKH[p, p]/(HDH[p, p]* HDH[p, p])
        
        #update classvector 
        for i in xrange(n_local):
            GOFMM_classes[rids[i], 0] = np.argmin(Similarity[i, :])+1
        print(GOFMM_classes.toArray())
        
        #update class indicator matrix H
        H_local = np.zeros([n_local, nclasses], dtype='float32', order='F')
        for i in xrange(n_local):
            H_local[i, <int>(GOFMM_classes[rids[i], 0] - 1)] = 1.0
        
        #copy class indicator matrix to DistData object
        H = PyGOFMM.PyDistData_RIDS(comm, N, nclasses, iset=rids, darr=H_local)

    classes = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, iset=gids)
    classes.redistribute(GOFMM_classes)

    return classes.toArray()

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
            ones = PyGOFMM.PyDistData_RIDS(self.comm, self.size, 1, arr=local_ones, iset=self.rids)
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
                normedx = np.zeros(np.shape(x), order='F', dtype='float32')
                for i in range(len(x)):
                    normedx[i] = 1/np.sqrt(self.D[self.rids[i], 0]) * x[i]
                GOFMM_x = PyGOFMM.PyDistData_RIDS(self.comm, m=self.size, n=nrhs, tree=self.K.getTree(), arr=normedx.astype('float32'))
                GOFMM_y = self.K.evaluate(GOFMM_x)
                for i in range(len(y)):
                    if self.norm:
                        #y[i] = 1/self.D[self.rids[i], 0] * GOFMM_y[self.rids[i], 0]
                        y[i] = 1/np.sqrt(self.D[self.rids[i], 0]) * GOFMM_y[self.rids[i], 0]
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
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    #setup PETSC context, normalized kernel D^-1K
    gofmm = GOFMM_Handler(K, norm=True)
    petsc4py.init(comm=comm)
    comm_petsc = MPI.COMM_WORLD
    A = PETSc.Mat().createPython([N, N], comm = comm_petsc)
    A.setPythonContext(gofmm)
    A.setUp()

    #don't search for the Perron-Frobenius vector
    x, b = A.createVecs()
    x.set(1.0)

    #setup nonhermitain eigenvalue problem
    E = SLEPc.EPS()
    E.create()
    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setFromOptions()
    E.setDimensions(nclasses)
    E.setDeflationSpace(x)
    E.solve()
    
    its = E.getIterationNumber()
    eps_type = E.getType()
    nev, ncv, mpd = E.getDimensions()
    tol, maxit = E.getTolerances()
    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    #get eigenvector solution
    eigvecs = []
    evals = []
    for i in range(nclasses-1):
        res = E.getEigenpair(i, vr, vi)
        eigvecs = eigvecs + [vr.copy()]
        evals = evals + [res.real]
        print(res.real)
    #turn eigenvectors into [nclasses x N] point cloud in reduced space
    spectral_points = np.empty([nclasses-1, n_local], dtype='float32', order='F')
    i = 0
    cdef float norm
    for vec in eigvecs:
        norm = vec.normalize()
        with vec as v:
            spectral_points[i, :] = v
        i = i + 1

    print(spectral_points)
    
    #Run K Means on Spectral Domain. Note these points are in RIDS/TREE ordering
    classes = KMeans(comm, spectral_points, nclasses, maxiter=20, init="++")
    #rearrange from rids->CBLK ordering. 
    classes = np.asarray(classes, dtype='float32')
    GIDS_Owned = PyGOFMM.getCBLKOwnership(N, rank, nprocs)
    CBLK_classes = PyGOFMM.PyDistData_RIDS(comm, N, 1, iset=GIDS_Owned)
    RIDS_classes = PyGOFMM.PyDistData_RIDS(comm, N, 1, iset=rids, arr=classes)
    CBLK_classes.redistribute(RIDS_classes)
    return CBLK_classes.toArray()
    #TODO: use the gofmm context syntax instead
    #GIDS_Owned = PyGOFMM.getCBLKOwnership(N, rank, nprocs)
    #spectral_sources = PyGOFMM.PyDistData_RIDS(comm, N, nclasses-1, iset=GIDS_Owned)
    #temp_sources = PyGOFMM.PyDistData_RIDS(comm, N, nclasses-1, iset=rids, darr=spectral_points)
    #spectral_sources.redistribute(temp_sources)
    #print(" ===== ")
    #CBLK_points = np.asarray(spectral_sources.toArray().T, order='F')
    #print(CBLK_points)
    #spectral_sourcesCBLK = PyGOFMM.PyDistData_CBLK(comm, nclasses-1, N, darr=CBLK_points)

    #config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.001, 0.01, False)
    #specK = PyGOFMM.KernelMatrix(comm, spectral_sourcesCBLK, conf=config, bandwidth = 0.1)
    
    #specK.compress()
    #return KKMeans(specK, nclasses, maxiter=20, gids=GIDS_Owned)

def DiffusionMap(PyGOFMM.KernelMatrix K, float eps, int[:] gids):
        
    cdef int N, n_local
    cdef int c, i, j, k, p, itr
    cdef MPI.Comm comm
    cdef int[:] rids = K.getTree().getGIDS().astype('int32') #get local row gofmm-tree ordering
    cdef float last_eig

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
    space = []
    
    last_eig = 1
    eigvecs = []
    eigvals = []
    while last_eig > eps:
        E = SLEPc.EPS()
        E.create()
        E.setOperators(A)
        E.setProblemType(SLEPc.EPS.ProblemType.HEP)
        E.setFromOptions()
        E.setDimensions(10)
        E.setDeflationSpace(space)
        E.solve()
        space = space + E.getInvariantSubspace()
        nconv = E.getConverged()
        vr, wr = A.getVecs()
        vi, wi = A.getVecs()
        for i in range(nconv):
            res = E.getEigenpair(i, vr, vi)
            eigvecs = eigvecs+[vr.copy()]
            eigvals = eigvals+[res.real]
            last_eig = res.real
            if last_eig < eps:
                break

    print(eigvals)
    #turn eigenvectors into [N x nvec] point cloud in reduced space
    nvec = len(eigvecs)
    spectral_points = np.empty([n_local, nvec], dtype='float32', order='F')
    i = 0
    for vec, val in zip(eigvecs, eigvals):
        with vec as v:
            print(np.abs(val)*v)
            spectral_points[:, i] = v*np.abs(val)
        i = i + 1

    #print(spectral_points)
    tempSpecDistData = PyGOFMM.PyDistData_RIDS(comm, N, nvec, darr=spectral_points, iset=rids)
    SpecDistData = PyGOFMM.PyDistData_RIDS(comm, N, nvec, iset=gids)
    SpecDistData.redistribute(tempSpecDistData)
    ##print(SpecDistData.toArray())
    return np.copy(SpecDistData.toArray())
    #return spectral_points



def MemoryTest(PyGOFMM.KernelMatrix K, int iterations):
        
    cdef int N, n_local
    cdef int c, i, j, k, p, itr
    cdef MPI.Comm comm
    cdef int[:] rids = K.getTree().getGIDS().astype('int32') #get local row gofmm-tree ordering
    cdef float last_eig

    N = K.getSize()
    n_local = len(rids)
    comm = K.getComm()

    local_ones = np.ones(n_local, dtype='float32', order='F')
    ones = PyGOFMM.PyDistData_RIDS(comm, N, 1, arr=local_ones, iset=rids)
    for i in xrange(iterations):
        K.evaluate(ones)
    
    #gofmm = GOFMM_Handler(K, norm=True)
    #petsc4py.init(comm=comm)
    #comm_petsc = MPI.COMM_WORLD
    #A = PETSc.Mat().createPython([N, N], comm = comm_petsc)
    #A.setPythonContext(gofmm)
    #A.setUp()
    #x, b = A.createVecs()
    #x.set(0.01)
    #for i in xrange(iterations):
    #    x = A*x
    #return x
