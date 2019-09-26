#Import from PyGOFMM
cimport pygofmm.core as PyGOFMM
import pygofmm.core as PyGOFMM #TODO: Why can't I see cdef declarations from core.pxd ?????

from pygofmm.petsc import *
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
import time

#named tuples
from collections import namedtuple

#plotting for debugging
import matplotlib.pyplot as plt

def gather_classes(comm, local_classes, gids_owned):
    recvbuf = comm.allgather(local_classes)
    classes = np.concatenate(recvbuf, axis=0).astype('int32')
    recvbuf = comm.allgather(gids_owned)
    gids = np.concatenate(recvbuf, axis=0).astype('int32')
    return (classes, gids)

def test_function(PyGOFMM.Runtime testparam):
    testparam.init()
    testparam.finalize()

#NOTE: Hey Sameer, check this out, apparently our toArray() isn't making a copy
#      using b afterwards is causing a segfault for come reason
def test_to_array(PyGOFMM.DistData_RIDS b):
    cdef float[:, :] t = b.to_array()
    print(t[0, 0])
    t[0, 0] = 1000
    print(t[0, 0])
    t = b.to_array()
    print(t[0, 0])
    print("Done")

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef compute_clustering_cost(MPI.Comm comm, float[:, :] points, int nclasses, float [:, :] centers):
    cdef float minimum_dist, temp_dist, local_sum, global_sum 
    cdef int n_local, d, r, i, j
    global_sum = 0.0
    n_local = np.shape(points)[1]
    d = np.shape(points)[0] 
    for r in prange(n_local, nogil=True):
        minimum_dist = -1.0
        for i in xrange(nclasses):
            temp_dist = 0.0
            for j in xrange(d):
                #TODO: Change this so its column major efficient?
                temp_dist = temp_dist + (points[r, j] - centers[i, j])* (points[r, j] - centers[i, j])
            if (temp_dist < minimum_dist) or (minimum_dist == -1):
                minimum_dist = temp_dist
        local_sum += minimum_dist

    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    return global_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float min_dist(int d, float[:] point, int nclasses, float[:, :] centers) nogil:
    cdef int i, j
    cdef float min_dist_value, temp_dist_value
    min_dist_value = -1.0
    for i in xrange(nclasses):
        temp_dist_value = 0.0
        for j in xrange(d):
            temp_dist_value = temp_dist_value +  (point[j] - centers[j, i])*(point[j] - centers[j, i])
        if (temp_dist_value < min_dist_value) or (min_dist_value == -1):
            min_dist_value = temp_dist_value
    return min_dist_value


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
            dlist[r] = min_dist(d, points[:, r], i, centerlist)

        cost = np.sum(dlist)
        dlist = local_prefix_sum(dlist)

        for r in prange(n_local, nogil=True):
            dlist[r] = dlist[r]/cost;

        u = np.random.rand()
        indx = local_bsearch(u, dlist)
        centerlist[:, i] = points[:, indx]
    
    return np.asarray(centerlist, dtype='float32', order='F')


cdef share_number(MPI.Comm comm, u, int r):
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
    u = share_number(comm, u, 0)
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
            dlist[r] = min_dist(d, points[:, r], k, centerlist)
    
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
        u = share_number(comm, u, 0)
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
    cdef double s_tinit, e_tinit, s_tup, e_tup, s_tcc, e_tcc
    cdef double center_time, update_time
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

    s_tinit = MPI.Wtime()

    if init == "random":
        for i in range(nclasses):
            u = 0
            if rank ==0:
                u = np.random.randint(0, nprocs)
            u = share_number(comm, u, 0)
            if rank == u:
                index = np.random.randint(0, n_local)
                temp_centroids[:, i] = points[:, index]
        comm.Allreduce(temp_centroids, centroids, op=MPI.SUM)
    elif init == "++":
        centroids = distributed_kmeans_pp(comm, points, nclasses)

    e_tinit= MPI.Wtime()

    center_time = 0
    update_time = 0
    cdef float distance, temp_dist
    for l in xrange(maxiter):
        #reassign classes
        s_tup = MPI.Wtime()
        update_classes(points, classes, centroids, nclasses)
        e_tup = MPI.Wtime()
        update_time += e_tup - s_tup
        #find centroids
        s_tcc = MPI.Wtime()
        centroids = compute_centers(comm, points, classes, nclasses)
        e_tcc = MPI.Wtime()
        center_time += e_tcc - s_tcc

    KMeansOutput = namedtuple('KMeansOutput', 'classes, init_time, center_time, update_time')
    output = KMeansOutput(np.asarray(classes), e_tinit -s_tinit, center_time, update_time) 
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef update_classes(float[:, :] points, int[:] classes, float[:, :] centroids, int nclasses=2):
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
cdef compute_centers(MPI.Comm comm, float[:, :] points, int[:] classes, int nclasses=2):
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





def computeConfusion(MPI.Comm comm, int[:] truth, int[:] classes, int nclass, int ncluster):
    cdef int n = len(truth)
    #assert len(truth) == len(classes)
    cdef int[:, :] confusion_local = np.zeros([ncluster+1, nclass+1], dtype='int32')
    for i in range(n):
        icluster = classes[i] - 1
        iclass = truth[i] - 1
        confusion_local[icluster, iclass] = confusion_local[icluster, iclass] + 1
    
    #Reduce confusion matrix
    cdef int [:, :] confusion = np.zeros([ncluster+1, nclass+1], dtype='int32')
    comm.Allreduce(confusion_local, confusion, op=MPI.SUM)

    for q in range(nclass):
        for p in range(ncluster):
            Cpq = confusion[p, q]
            confusion[p, nclass] = confusion[p, nclass] + Cpq
            confusion[ncluster, q] = confusion[ncluster, q] + Cpq
            confusion[ncluster, nclass] = confusion[ncluster, nclass] + Cpq
    
    #confusion_local = np.copy(confusion)
    #confusion = np.zeros([ncluster+1, nclass+1], dtype='int32')

    #comm.Allreduce(confusion_local, confusion, op=MPI.SUM)

    return confusion


def ChenhanNMI(MPI.Comm comm, int[:] truth, int[:] classes, int nclass, int ncluster):
    nmi = 0.0;
    nmi_a = 0.0;
    nmi_c = 0.0;

    n_local = np.array(len(truth), dtype='int32')
    n = np.array(0, dtype='int32')
    comm.Allreduce(n_local, n, op=MPI.SUM)
    
    #assert len(truth) == len(classes)

    confusion = computeConfusion(comm, truth, classes, nclass, ncluster)

    print(np.asarray(confusion))

    #antecendent part
    for q in range(nclass):
        for p in range(ncluster):
            Cpq = confusion[p, q];
            Cp = confusion[p, nclass]
            Cq = confusion[ncluster, q]

            if Cpq > 0.0:
                nmi_a += -2 * (Cpq/n) * log2(n*Cpq/(Cp*Cq))

    #consequent part
    for q in range(nclass):
        Cq = confusion[ncluster, q]
        nmi_c += (Cq/n) * log2(Cq/n)

    for p in range(ncluster):
        Cp = confusion[p, nclass];
        nmi_c += (Cp/n) * log2(Cp/n)

    local_nmi_a = np.array(nmi_a)
    local_nmi_c = np.array(nmi_c)
    nmi_a = np.array(nmi_a)
    nmi_c = np.array(nmi_c)

    comm.Allreduce(local_nmi_a, nmi_a, op=MPI.SUM)
    comm.Allreduce(local_nmi_c, nmi_c, op=MPI.SUM)

    print(nmi_a)
    print(nmi_c)

    return nmi_a/nmi_c
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def NMI(MPI.Comm comm, int[:] truth, int[:] clusters, int nclasses):
    #Compute entropy of true labelings
    cdef double start_time, end_time
    cdef int i, j, k, c
    cdef float t, h_y, h_c
    start_time = MPI.Wtime()
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
        
    cdef float[:] cluster_counts_local = np.zeros([nclasses], dtype='float32')
    cdef float[:] cluster_counts = np.copy(truth_counts_local)
    for i in xrange(n_local):
        c = clusters[i]-1
        cluster_counts_local[c] += 1

    comm.Allreduce(cluster_counts_local, cluster_counts, op=MPI.SUM)
    for j in xrange(nclasses):
        t = cluster_counts[j]/N_total
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
            t = 0
            if cluster_counts[i]:
                t = counts[i, j]/cluster_counts[i]
            if t>0:
                I[i] += t*log2(t)
    cdef float h_yc = 0.0
    for i in xrange(nclasses):
        h_yc -= cluster_counts[i]/N_total * I[i]
    end_time = MPI.Wtime()

    print(end_time)
    return ( (2*(h_y - h_yc)/(h_y + h_c)), end_time - start_time)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def KernelKMeans(PyGOFMM.KernelMatrix K, int nclasses, gids, classvec=None, maxiter=10, init="random"):
    cdef int N, n_local, m, d
    cdef int c, i, j, k, p, itr, u
    cdef MPI.Comm comm
    cdef int[:] rids = K.get_tree().get_gids().astype('int32')

    N = K.get_size()
    n_local = len(rids)
    comm = K.get_comm()
    nprocs = comm.size
    rank = comm.rank

    cdef float[:] centers = np.zeros([nclasses], dtype='float32')
    cdef int[:] center_ind = np.zeros([nclasses], dtype='int32')
    cdef int[:] center_ind_local = np.zeros([nclasses], dtype='int32')
    cdef float[:] dlist = np.zeros([n_local], dtype='float32')

    cdef float min_dist_value, cur_dist_value
    cdef int start, end

    #form density weights D
    cdef float[:, :] local_ones = np.ones([n_local, 1], dtype='float32', order='F')
    Ones = PyGOFMM.DistData_RIDS(comm, N, 1, iset=rids, darr=local_ones)
    D = K.evaluate(Ones)

    #create local numpy copies to use shared memory parallelism 
    cdef float[:] npD = D.to_array().flatten()

    if classvec is None and init == "random":
        classvec = np.zeros([n_local], order='F', dtype='float32')
        for i in xrange(n_local):
            classvec[i] = np.random.randint(1, nclasses+1)

    if classvec is None and init =="random_points":
        t = 1000 * MPI.Wtime() # current time in milliseconds
        np.random.seed(int(t) % 2**32)
        if rank < nclasses % nprocs:
            start = rank*(nclasses/nprocs + 1)
            end = (rank + 1)*(nclasses/nprocs +1)
        else:
            start = (rank - nclasses % nprocs)*(nclasses/nprocs) + (nclasses % nprocs) * (nclasses/nprocs + 1)
            end = start + (nclasses/nprocs)
        
        print(start)
        print(end)
        sys.stdout.flush()
        
        for i in range(start, end):
            center_ind_local[i] = np.random.randint(0, n_local)

        comm.Allreduce(center_ind_local, center_ind, op=MPI.SUM)
        classvec = np.zeros([n_local], dtype='float32')
        for j in xrange(n_local):
            min_dist_value = -1;
            for k in xrange(nclasses):
                cind = center_ind[k]
                cur_dist_value = 2*npD[i] - 2*npD[cind]*K[cind, j]
                sys.stdout.flush()
                if(cur_dist_value < min_dist_value) or (min_dist_value== -1):
                    min_dist_value = cur_dist_value
                    c = k
        
            classvec[j] = c + 1

    if classvec is None and init=="pp":
        t = 1000 * MPI.Wtime()
        np.random.seed(int(t) %2**32)

        #create buffer for center indicator matrix
        I = PyGOFMM.DistData_RIDS(comm, m = N, n=nclasses, iset=rids)
        npI = I.to_array() #NOTE: Be CAREFUL with scope (this is not a copy)

        #create random sequence
        if rank == 0:
            sequence = np.random.rand(nclasses+1)
        sequence = comm.bcast(sequence, root=0)

        #picking initial point
        #choose a random processor
        p = <int>(np.floor(sequence[0]*nprocs))

        if rank == p:
            #choose a random point and set indicators
            ind = <int>(np.floor((sequence[0] - rank/nprocs) * n_local * nprocs))
            print(ind)
            npI[ind, 0] = npD[ind]

        dlists = K.evaluate(I)
        dlists = -1 * dlists.to_array()

        #Start Main Loop
        k = 1
        dlist = np.amin(dlists[:, :k], axis=1)
        dlist = np.add(npD, dlist)

        local_cost = np.array(np.sum(dlist), dtype='float32')
        dlist = local_prefix_sum(dlist)

        #compute global cost
        cost = np.array(0.0, dtype='float32')
        comm.Barrier()
        comm.Allreduce(local_cost, cost, op=MPI.SUM)

        #compute plist (to determine which rank to sample from)
        sendbuf = np.zeros([1], dtype='float32') + local_cost
        recvbuf = np.zeros([nprocs], dtype='float32')
        recvbuf = comm.allgather(sendbuf)
        
        plist = np.concatenate(recvbuf)
        plist = local_prefix_sum(plist)
        plist = np.asarray(plist)/cost
        
        #sample based on local cost per rank
        indx_g = local_bsearch(sequence[1], plist)

        if rank == indx_g:
            dlist = np.asarray(dlist)/local_cost
            indx = local_bsearch(sequence[1], dlist)
            print(indx)
            npI[indx, k] = npD[indx]

        dlists = K.evaluate(I)
        dlists = dlists.to_array()
        dlists = -1 * dlists
        
        #End main loop

        #Loop through final dlists to assign points
        classvec = np.zeros([n_local], dtype='float32')
        for i in range(n_local):
            c = amin(dlists[i, :], 2) + 1
            classvec[i] = c

    #Redistribute classvec to rids
    #classes = PyGOFMM.DistData_RIDS(comm, N, 1, iset=gids, arr=classvec)
    #GOFMM_classes = PyGOFMM.DistData_RIDS(comm, m=N, n=1, iset=rids)
    #GOFMM_classes.redistribute(classes)
    GOFMM_classes = PyGOFMM.DistData_RIDS(comm, m=N, n=1, iset=rids, arr=classvec)

    #initialize class indicator block
    cdef float[:, :] H_local = np.zeros([n_local, nclasses], dtype='float32', order='F')
    for i in xrange(n_local):
        H_local[i, <int>(GOFMM_classes[rids[i], 0] -1)] = 1.0

    print("Made class indicator block")
    sys.stdout.flush()
    #copy class indicator block to DistData object
    H = PyGOFMM.DistData_RIDS(comm, N, nclasses, iset=rids, darr=H_local)

    #create local numpy copies to use shared memory parallelism 
    cdef float[:] Diag = np.ones([n_local], dtype='float32', order='F') 
    
    #allocate storage for lookup matricies
    cdef float[:, :] HKH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] HKH = np.zeros([nclasses, nclasses], dtype='float32', order='F')

    cdef float[:, :] HDH_local = np.zeros([nclasses, nclasses], dtype='float32', order='F')
    cdef float[:, :] HDH = np.zeros([nclasses, nclasses], dtype='float32', order='F')

    cdef float[:, :] DKH = np.zeros([n_local, nclasses], dtype='float32', order='F')

    cdef float[:, :] Similarity = np.zeros([n_local, nclasses], dtype='float32', order='F')
    cdef float[:, :] npKH
    cdef float[:, :] npH

    cdef int[:] class_counts = np.zeros([nclasses], dtype='int32', order='F')

    cdef int r
    cdef float[:, :] c_classes

    #start main loop
    cdef float sim_time = 0;
    cdef float np_time = 0;
    cdef float matvec_time = 0;
    cdef float com_time = 0;
    cdef float nrmlz = 0

    for itr in xrange(maxiter):

        #update DKH, HKH, HDH
        time = MPI.Wtime()

        #GOFMM evaluate 
        KH = K.evaluate(H)

        #create local numpy copies to use shared memory parallelism
        npKH = KH.to_array();
        npH = H.to_array();

        HDH_local = np.zeros([nclasses, nclasses], dtype='float32')
        HKH_local = np.zeros([nclasses, nclasses], dtype='float32')

        #TODO: Replace with shared memory parallel or MKL/BLAS
        for i in xrange(nclasses):
            for j in xrange(nclasses):
                for r in prange(n_local, nogil=True):
                    HKH_local[i, j] += npH[r, j]*npKH[r, i]
                    HDH_local[i, j] += npH[r, j]*npD[r]*npH[r, i]

        print(np.asarray(HKH_local))
        print(np.asarray(HDH_local))

        HDH_local = np.zeros([nclasses, nclasses], dtype='float32')
        HKH_local = np.zeros([nclasses, nclasses], dtype='float32')

        #TODO: Replace with shared memory parallel or MKL/BLAS
        for i in xrange(nclasses):
            for j in xrange(nclasses):
                for r in range(n_local):
                    HKH_local[i, j] += npH[r, j]*npKH[r, i]
                    HDH_local[i, j] += npH[r, j]*npD[r]*npH[r, i]

        print(np.asarray(HKH_local))
        print(np.asarray(HDH_local))

        for i in range(n_local):
            for j in xrange(nclasses):
                DKH[i, j] = 1/npD[i] * npKH[i, j]

        HKH = np.zeros([nclasses, nclasses], dtype='float32', order='F')
        HDH = np.zeros([nclasses, nclasses], dtype='float32', order='F')

        comm.Allreduce(HKH_local, HKH, op=MPI.SUM)
        comm.Allreduce(HDH_local, HDH, op=MPI.SUM)

        np_time += MPI.Wtime() - time
        
        time = MPI.Wtime()

        #update similarity
        for i in range(n_local):
            for p in xrange(nclasses):
                #Similarity[i, p] = Diag[i]/(npD[i]*npD[i]) - 2*DKH[i, p]/HDH[p, p] + HKH[p, p]/(HDH[p, p]*HDH[p, p])
                nrmlz = HDH[p, p]
                if nrmlz == 0:
                    Similarity[i, p] = 100
                    print("Skipping this")
                else:
                    Similarity[i, p] = -2*(DKH[i, p]/nrmlz) + (HKH[p, p]/(nrmlz*nrmlz))


        #update classvector
        c_classes = GOFMM_classes.to_array()
        for i in range(n_local):
            c_classes[i, 0] = amin(Similarity[i, :], nclasses)+1

        #reset class counts
        for i in range(nclasses):
            class_counts[i] = 0
    
        #count classes
        for i in range(n_local):
            c = <int>(c_classes[i, 0]) - 1
            class_counts[c] = <int>(class_counts[c]) + 1

        print(np.asarray(class_counts))
        for c in range(nclasses):
            if class_counts[c] == 0:
                print("Randomly adding point")
                #change random point to class c
                ind = np.random.randint(0, n_local)
                c_classes[ind, 0] = <float>(c) + 1

        #TODO: Change this to just updaing npH, don't need to allocate new space 
        #update class indicator matrix H
        H_local = np.zeros([n_local, nclasses], dtype='float32', order='F')
        for i in range(n_local):
            H_local[i, <int>(c_classes[i, 0] -1)] = 1.0

        #copy class indicator matrix to DistData object
        H = PyGOFMM.DistData_RIDS(comm, N, nclasses, iset=rids, darr=H_local)
        sim_time += MPI.Wtime() - time

    #end main loop

    time = MPI.Wtime()
    classes = PyGOFMM.DistData_RIDS(comm, m=N, n=1, iset=gids)
    classes.redistribute(GOFMM_classes)
    com_time = MPI.Wtime() - time
    KMeansOutput = namedtuple('KMeansOutput', 'classes, matvec_time, numpy_time, similarity_time, communication_time')
    output = KMeansOutput(classes.to_array(), matvec_time, np_time, sim_time, com_time)
    return output
        

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int amin(float[:] arr, int n) nogil:
    cdef float min_value
    cdef int i, j, indx
    indx = 0
    min_value = -1
    for i in xrange(n):
        if (arr[i] < min_value) or (min_value ==-1):
            min_value = arr[i]
            indx = i
    return indx


# Get vector for KDE multiply
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def KDE_Vec( MPI.Comm comm,int N, int nclasses, float[:] classvec, int[:] user_rids, int[:] tree_rids):
    cdef int n_local = len(tree_rids)

    # redistribute
    classes_user = PyGOFMM.DistData_RIDS(comm, m=N, n=1, arr=classvec, iset=user_rids)
    cdef PyGOFMM.DistData_RIDS classes_hmlp = PyGOFMM.DistData_RIDS(comm, m=N, n=1, iset=tree_rids)
    classes_hmlp.redistribute(classes_user)

    #initialize class indicator block
    cdef float[:, :] ww_hmlp_loc= np.zeros([n_local, nclasses], dtype='float32', order='F')
    #for i in xrange(n_local): #TODO fix loop?
    #    ww_hmlp_loc[i, <int>(classes_hmlp[tree_rids[i], 0] - 1)] = 1.0

    #copy class indicator block to DistData object
    cdef PyGOFMM.DistData_RIDS bla = PyGOFMM.DistData_RIDS(comm, N, nclasses, iset=tree_rids, darr=ww_hmlp_loc)
    return bla


#pure python implementation of KDE
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def KDE(PyGOFMM.KernelMatrix K, int nclasses,int[:] gids, float[:] classvec):
    cdef int N, n_local
    cdef int i,ci,j
    cdef MPI.Comm comm
    cdef int[:] rids = K.get_tree().get_gids().astype('int32') #get local row gofmm-tree ordering
    
    N = K.get_size()
    n_local = len(rids)
    comm = K.get_comm()

    #load class data into PyGOFMM DistData object, TODO: necessary for creation of ww_hmlp?
    vec_start = time.time()
    #classes_user = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, arr=classvec, iset=gids)
    #classes_hmlp = PyGOFMM.PyDistData_RIDS(comm, m=N, n=1, iset=rids)
    #classes_hmlp.redistribute(classes_user)

    ##initialize class indicator block
    #cdef float[:, :] ww_hmlp_loc= np.zeros([n_local, nclasses], dtype='float32', order='F')
    #for i in xrange(n_local): #TODO fix loop?
    #    ww_hmlp_loc[i, <int>(classes_hmlp[rids[i], 0] - 1)] = 1.0

    ##copy class indicator block to DistData object
    #cdef PyGOFMM.PyDistData_RIDS ww_hmlp = PyGOFMM.PyDistData_RIDS(comm, N, nclasses, iset=rids, darr=ww_hmlp_loc)
    cdef PyGOFMM.DistData_RIDS ww_hmlp = KDE_Vec(comm, N, nclasses, classvec, gids, rids)
    vec_end = time.time()
    vec_time =vec_end - vec_start

    # Compute multiply
    kde_start = time.time()
    cdef PyGOFMM.DistData_RIDS density_hmlp = K.evaluate(ww_hmlp)

    # subtract self interactions (TODO)
    #density_hmlp.Subtract(ww_hmlp)

    # Redistribute
    cdef PyGOFMM.DistData_RIDS density_user = PyGOFMM.DistData_RIDS(comm,m = N,n=nclasses, iset = gids)
    density_user.redistribute(density_hmlp)
    kde_end = time.time()
    kde_time = kde_end - kde_start

    print("  KDE event  |   Time")
    print(" ---------------------")
    print(" vec create  |   ",vec_time)
    print("   run kde   |   ",kde_time)
    sys.stdout.flush() 

    # output density 
    return density_user.to_array() # - ww_hmlp_loc #TODO current default removes self interactions. keep?

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def TestKDE(PyGOFMM.KernelMatrix K, int nclasses, int [:] gids, float[:] classvec, float[:,:] Xte):
    cdef int N, n_local
    cdef int i,ci,j
    cdef MPI.Comm comm
    cdef int[:] rids = K.get_tree().get_gids().astype('int32') #get local row gofmm-tree ordering
    
    N = K.get_size()
    n_local = len(rids)
    comm = K.get_comm()

    #load class data into PyGOFMM DistData object, TODO: necessary for creation of ww_hmlp?
    classes_user = PyGOFMM.DistData_RIDS(comm, m=N, n=1, arr=classvec, iset=gids)
    classes_hmlp = PyGOFMM.DistData_RIDS(comm, m=N, n=1, iset=rids)
    classes_hmlp.redistribute(classes_user)

    #initialize class indicator block
    cdef float[:, :] ww_hmlp_loc= np.zeros([n_local, nclasses], dtype='float32', order='F')
    for i in xrange(n_local): #TODO fix loop? This can be parallelized
        ww_hmlp_loc[i, <int>(classes_hmlp[rids[i], 0] - 1)] = 1.0

    #copy class indicator block to DistData object
    ww_hmlp = PyGOFMM.DistData_RIDS(comm, N, nclasses, iset=rids, darr=ww_hmlp_loc)

    # Compute multiply
    cdef PyGOFMM.LocalData Xte_py = PyGOFMM.LocalData( m = Xte.shape[0], n = Xte.shape[1], darr = Xte)
    density_test = K.evaluateTest(Xte_py, ww_hmlp)

    # return
    return density_test.to_array()

def SpectralCluster(PyGOFMM.KernelMatrix K, int nclasses, int[:] gids, init="random"):
        
    cdef int N, n_local
    cdef int c, i, j, k, p, itr
    cdef MPI.Comm comm
    cdef int[:] rids = K.get_tree().get_gids().astype('int32') #get local row gofmm-tree ordering
    cdef double stime, etime
    cdef double eig_time

    N = K.get_size()
    n_local = len(rids)
    comm = K.get_comm()
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    #setup PETSC context, normalized kernel D^-1K
    gofmm = Kernel_Handler(K, normalize=True)
    petsc4py.init(comm=comm)
    comm_petsc = MPI.COMM_WORLD
    A = PETSc.Mat().createPython(( (n_local, N), (n_local, N) ), comm = comm_petsc)
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
    E.setDimensions(nclasses+1) #Currently getting one more than needed to see if it helps with clustering
    E.setDeflationSpace(x)
    stime = MPI.Wtime()
    E.solve()
    etime = MPI.Wtime()
    eig_time= stime - etime
    
    its = E.getIterationNumber()
    eps_type = E.getType()
    nev, ncv, mpd = E.getDimensions()
    tol, maxit = E.getTolerances()
    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()


    print("Number of Converged Eigenvalues:", nconv)
    print("Number of iterations (bugged in petsc4py)", its)
    #get eigenvector solution
    eigvecs = []
    evals = []
    for i in range(nclasses):
        res = E.getEigenpair(i, vr, vi)
        eigvecs = eigvecs + [vr.copy()]
        evals = evals + [res.real]
        print(res.real)

    #turn eigenvectors into [nclasses x N] point cloud in reduced space
    #normalize for spherical k-means
    spectral_points = np.empty([nclasses, len(rids)], dtype='float32', order='F')
    i = 0
    cdef float norm
    for vec in eigvecs:
        norm = vec.normalize()
        with vec as v:
            spectral_points[i, :] = v/norm
        i = i + 1
    
    #Run K Means on Spectral Domain. Note these points are in RIDS/TREE ordering
    output  = KMeans(comm, spectral_points, nclasses, maxiter=40, init=init)
    classes = output.classes
    center_time = output.center_time
    update_time = output.update_time
    init_time = output.init_time

    #rearrange from rids->CBLK ordering. 
    classes = np.asarray(classes, dtype='float32')
    GIDS_Owned = PyGOFMM.get_cblk_ownership(N, rank, nprocs)
    CBLK_classes = PyGOFMM.DistData_RIDS(comm, N, 1, iset=GIDS_Owned)
    RIDS_classes = PyGOFMM.DistData_RIDS(comm, N, 1, iset=rids, arr=classes)
    CBLK_classes.redistribute(RIDS_classes)

    SpectralClusterOutput= namedtuple("SpectralClusterOutput", 'classes, eigensolver_time, center_time, update_time, init_time, rids_points, rids_classes')
    output = SpectralClusterOutput(CBLK_classes.to_array(), eig_time, center_time, update_time, init_time, spectral_points, classes)
    return output

def DiffusionMap(PyGOFMM.KernelMatrix K, float eps, int[:] gids):
        
    cdef int N, n_local
    cdef int c, i, j, k, p, itr
    cdef MPI.Comm comm
    cdef int[:] rids = K.get_tree().get_gids().astype('int32') #get local row gofmm-tree ordering
    cdef float last_eig
    cdef double stime, etime, eig_time

    N = K.get_size()
    n_local = len(rids)
    comm = K.get_comm()

    gofmm = Kernel_Handler(K, normalize=True)
    petsc4py.init(comm=comm)
    comm_petsc = MPI.COMM_WORLD
    A = PETSc.Mat().createPython([N, N], comm = comm_petsc)
    A.setPythonContext(gofmm)
    A.setUp()
    x, b = A.createVecs()
    x.set(1.0)
    space = []
    stime = MPI.Wtime()
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

    etime = MPI.Wtime()
    eig_time = etime - stime
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
    tempSpecDistData = PyGOFMM.DistData_RIDS(comm, N, nvec, darr=spectral_points, iset=rids)
    SpecDistData = PyGOFMM.DistData_RIDS(comm, N, nvec, iset=gids)
    SpecDistData.redistribute(tempSpecDistData)
    ##print(SpecDistData.to_array())
    return (np.copy(SpecDistData.to_array()), eig_time)



def MemoryTest(PyGOFMM.KernelMatrix K, int iterations):
        
    cdef int N, n_local
    cdef int c, i, j, k, p, itr
    cdef MPI.Comm comm
    cdef int[:] rids = K.get_tree().get_gids().astype('int32') #get local row gofmm-tree ordering
    cdef float last_eig

    N = K.get_size()
    n_local = len(rids)
    comm = K.getComm()

    local_ones = np.ones(n_local, dtype='float32', order='F')
    ones = PyGOFMM.DistData_RIDS(comm, N, 1, arr=local_ones, iset=rids)
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
