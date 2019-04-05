#from PyRuntime import *
#from PyMatrix import *
import PyGOFMM_Dist as pyg
from mpi4py import MPI
import numpy as np
import IPython


def LocalCount(crank,csize,ntot):
    n_per = (int) (ntot/csize)
    n_mod = (int) (ntot % csize)
    n_start = (int) (crank * n_per)
    return n_start,n_per

    #if (crank > n_mod):
    #    n_start += crank
    #    n_per += 1
    #    return n_start,n_per
    #else:
    #    return n_start,n_per

def makeCBLKIndex(crank, csize, ntot):
    cblk_idx = np.arange(crank,ntot, csize)
    return cblk_idx

# data creation func
def CreateLocalData(ntot, crank, csize, sep = 5, classes = 2):
    # create global array 
    class_1 = np.random.randn(d, (int)(np.floor(ntot/classes)))+sep
    class_2 = np.random.randn(d, (int)(np.ceil(ntot/classes)))
    local_data = np.concatenate((class_1, class_2), axis=1) 

    # extract my section
    #n_start,n_per = LocalCount(crank,csize,ntot)
    #local_data = local_data[:, n_start:(n_start  + n_per)]
    cblk_idx = makeCBLKIndex(crank,csize,ntot)
    local_data = local_data[:,cblk_idx]

    # ensure fortran ordering
    local_data = np.asfortranarray(local_data.astype('float32'))
    return local_data

# data create from file
def LoadLocalData(ntot,crank,csize,filename):
    # load full array
    full_array = np.fromfile(fname,dtype='float32')
    full_array = np.reshape(full_array,(-1,ntot),order='F')

    print(" Val should be > 2: ", full_array[0,3000])

    # extract my section
    #n_start, n_per = LocalCount(crank,csize,ntot)
    #local_data = full_array[:,n_start:(n_start + n_per)]
    cblk_idx = makeCBLKIndex(crank,csize,ntot)
    local_data = full_data[:,cblk_idx]

    # ensure fortran ordering
    local_data = np.asfortranarray(local_data.astype('float32'))
    return local_data

# create identity columns, and note index
def CreateRHSLocal(rids,ntot,nrhs = 10):
    n_sep = (int)(ntot/nrhs)
    rhs = np.zeros([ntot,nrhs])
    id_idx = np.zeros(nrhs)

    for ri in range(nrhs):
       ri_cur = (int) (n_sep * ri)
       rhs[ri_cur, ri] = 1.0
       id_idx[ri] = ri_cur 

    rhs = rhs[rids,:]
    rhs = np.asfortranarray(rhs.astype('float32'))

    return rhs,id_idx


def PrintIdentityErrors(vec_out, true_id_idx, my_rids):
    nrhs = vec_out.cols()
    for ri in range(nrhs):
        cur_ri = true_id_idx[ri]
        if cur_ri in my_rids:
            # should be 1
            print(" Value for (", cur_ri, ",",ri,") is ", vec_out[cur_ri,ri])
    
            # test another random value 
            new_ri =(int)( (ri + nrhs/2) % nrhs)
            print(" Value for (", cur_ri, ",",true_id_idx[new_ri],") is ", uu[cur_ri,new_ri],"\n")


def PrintKernelInfo(Kmat, true_id_idx, crank):
    nrhs = len(true_id_idx)
    print("---- printing kernel info on rank ",crank,"----")
    for ri in range(nrhs):
        cur_ri = true_id_idx[ri]
        opp_ri = true_id_idx[ (int) ((ri + nrhs/2) % nrhs) ]
        kval = KK.getValue(cur_ri,opp_ri)

        print(" Value at (",cur_ri,",",opp_ri,") is ", kval) 

"""
Beginning of python real script to test gofmm multiply
Section 0: Common parameters
"""
# mpi stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

prt = pyg.PyRuntime()
prt.init_with_MPI(comm)

# parameters
n = 4000 # number of points
d = 2 # dimensionality of data
ktype = 'GAUSSIAN' # kernel type
print_rank = 1 # rank to print from
m = 128
k = 64
s = 128 
stol = 1e-5
budget = 0.01
nrhs = 10
fname = "../bin/points.bin"

# create local data
#np.random.seed(10)
#loc_data = CreateLocalData(n,rank,size)
loc_data = LoadLocalData(n,rank,size,fname)
Xtr = pyg.PyDistData_CBLK(comm,m = d, n = n, darr=loc_data)

"""
Section 1: Simple matvec -- c++ wrapped with only python
"""
# initialize config
conf = pyg.PyConfig("GEOMETRY_DISTANCE",n,m,k,s,stol,budget,True)

# Kernel matrix
KK = pyg.KernelMatrix(comm,Xtr,conf=conf)

# compress
bla1 = KK.getValue(400,800)
KK.compress()
bla2 = KK.getValue(400,800)
print(" Before: ", bla1, " After: ",bla2)

# Create multiply vector
rids = KK.getTree().getGIDS()
rhs_local,id_idx = CreateRHSLocal(rids,n,nrhs)
#rhs_local = np.asfortranarray(np.ones([len(rids),nrhs]).astype('float32'))

# load into distdata and multiply
ww = pyg.PyDistData_RIDS(comm,m = n, n = nrhs,iset=rids, darr=rhs_local)
#ww_rblk = pyg.PyDistData_RBLK(comm,m = n, n = 1, darr=rhs_local)
#ww = pyg.PyDistData_RIDS(comm,m=n,n=1,iset = rids)
#ww.loadRBLK(ww_rblk)
uu = KK.evaluate(ww)

# Output whether all expected values are 1.0
PrintIdentityErrors(uu, id_idx,rids)
if 2031 in rids:
    bla = np.where(rids == 2031)
    lid = (int) (bla[0][0])
    uu_npy = uu.toArray()
    print(" U value at 2031    : ", uu[2031,0])
    print(" U value at 2031(v) : ", uu.vecAccess(lid))
    print(" U value at 2031(np): ", uu_npy[lid])
PrintKernelInfo(KK,id_idx,rank)

"""
Section 2: Simple matvec -- c++ wrapped through petsc4py
"""


prt.finalize()
