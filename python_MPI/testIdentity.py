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


# data creation func
def CreateLocalData(ntot, crank, csize, sep = 5, classes = 2):
    # create global array 
    class_1 = np.random.randn(d, (int)(np.floor(ntot/classes)))+sep
    class_2 = np.random.randn(d, (int)(np.ceil(ntot/classes)))
    local_data = np.concatenate((class_1, class_2), axis=1) 

    # extract my section
    n_start,n_per = LocalCount(crank,csize,ntot)
    local_data = local_data[:, n_start:(n_start  + n_per)]

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
            new_ri = (ri + nrhs/2) % nrhs
            print(" Value for (", cur_ri, ",",new_ri,") is ", uu[cur_ri,new_ri],"\n")


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
d = 3 # dimensionality of data
ktype = 'GAUSSIAN' # kernel type
print_rank = 1 # rank to print from
m = 128
k = 64
s = 128 
stol = 1e-5
budget = 0.01
nrhs = 10

# create local data
np.random.seed(10)
loc_data = CreateLocalData(n,rank,size)
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

# Multiply by identity subset
rids = KK.getTree().getGIDS()
rhs_local,id_idx = CreateRHSLocal(rids,n)
ww = pyg.PyDistData_RIDS(comm,m = n, n = nrhs,iset=rids, darr=rhs_local)
#rhs_local = np.asfortranarray(np.ones([len(rids),1]).astype('float32'))
#ww = pyg.PyDistData_RIDS(comm,m = n, n = 1,iset=rids, darr=rhs_local)
uu = KK.evaluate(ww)

# Output whether all expected values are 1.0
PrintIdentityErrors(uu, id_idx,rids)
#if 2031 in rids:
#    print(" U value : ", uu[2031,0])

"""
Section 2: Simple matvec -- c++ wrapped through petsc4py
"""


prt.finalize()
