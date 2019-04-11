import PyGOFMM_Dist as pyg
from mpi4py import MPI
import numpy as np
import IPython

# CBLK index function
def makeCBLKIndex(crank, csize, ntot):
    cblk_idx = np.arange(crank,ntot, csize).astype('int32')
    return cblk_idx

def LocalCount(crank,csize,ntot):
    n_per = (int) (ntot/csize)
    n_mod = (int) (ntot % csize)
    n_start = (int) (crank * n_per)
    return n_start,n_per

# Chunk index function
def makeCHUNKIndex(crank,csize,ntot):
    n_start, n_per = LocalCount(crank,csize,ntot)
    chunk_idx = np.arange(n_start, (n_start + n_per)).astype('int32')
    return chunk_idx

# get rids 
def GetUserRIDS(crank,csize,ntot,form='cyclic'):
    if form is 'chunked':
        sub_idx = makeCHUNKIndex(crank,csize,ntot) 
    elif form is 'cyclic':
        sub_idx = makeCBLKIndex(crank,csize,ntot)
    else:
        raise Exception("Unknown user rids type")

    return sub_idx

# data create from file
def LoadLocalData(ntot,crank,csize,filename):
    # load full array
    full_array = np.fromfile(fname,dtype='float32')
    full_array = np.reshape(full_array,(-1,ntot),order='F')

    # extract my section
    cblk_idx = makeCBLKIndex(crank,csize,ntot)
    local_data = full_array[:,cblk_idx]

    # ensure fortran ordering
    local_data = np.asfortranarray(local_data.astype('float32'))
    return local_data

# load local classes
def LocalClasses(ntot,classes,crank,csize,form='cyclic'):
    # form full global list of class ids
    class_ids = np.ones( ntot,order='F' )
    class_ids[ (int)(ntot/2): (int)(ntot) ]  = 2

    # extract local from class ids
    sub_idx = GetUserRIDS(crank,csize,ntot,form)
    loc_ids = class_ids[sub_idx]

    # class list
    class_list = np.asfortranarray([1,2])

    return loc_ids,class_list

# make kde vec 
def KDEVec(loc_ids,class_list):
    # make vector from loc ids
    loc_kde_vec = np.asfortranarray( np.zeros( (loc_ids.size, class_list.size) ).astype('float32') )
    for ci in range(class_list.size):
        loc_kde_vec[ loc_ids == class_list[ci],ci ] = 1.0

    return loc_kde_vec


# Redistribute vector 
def VecRedistribute(win, mcomm, form = 'cyclic', tree_rids = None):
    if tree_rids is not None and form is 'tree':
        out_rids = tree_rids
    elif form is 'cyclic' or form is 'chunked':
        crank = mcomm.Get_rank()
        csize = mcomm.Get_size()
        ntot = win.rows()
        out_rids = GetUserRIDS(crank,csize,ntot,form)
    else:
        raise Exception("Unknown vector form for redistribute")

    wout = pyg.PyDistData_RIDS(comm, win.rows(), win.cols(), iset = out_rids)
    wout.redistribute(win)

    return wout
    

# KDE manager
def KDE(comm,KK,class_ids,class_list,form='cyclic'):



    return density




"""
[1.] Initialize environment, variables, and communications
"""
comm = MPI.COMM_WORLD
prt = pyg.PyRuntime()
prt.init_with_MPI(comm)
rank = comm.Get_rank()
size = comm.Get_size()

# GOFMM params
n = 4000 # number of points
d = 2 # dimensionality of data
ktype = 'GAUSSIAN' # kernel type
m = 128
k = 64
s = 128 
stol = 1e-5
budget = 0.01
sec_acc = True # whether to secure accuracy

# KDE params
fname = "../bin/points.bin"
bandwidth = 0.5
classes = 2
user_form = 'cyclic'

"""
[2.] Load data
    - assumed to be local sections of global data
    - either stored cyclically by rank, or in chunks
"""
loc_data = LoadLocalData(n,rank,size,fname)
Xtr = pyg.PyDistData_CBLK(comm,m = d, n = n, darr=loc_data)
class_ids,class_list = LocalClasses(n,classes,rank,size,form=user_form)

"""
[3.] Initialize kernel matrix and multiply vec
"""
# initialize config
conf = pyg.PyConfig("GEOMETRY_DISTANCE",n,m,k,s,stol,budget,sec_acc)

# Kernel matrix
KK = pyg.KernelMatrix(comm,Xtr,conf=conf,bandwidth = bandwidth)

# compress
KK.compress()

"""
[4.] Call kde 
"""
## TODO: all this belongs in KDE
# Create multiply vector
rhs_local = KDEVec(class_ids,class_list)
rhs_rids = GetUserRIDS(rank,size,n,user_form)
ww_user = pyg.PyDistData_RIDS(comm,m = n, n = classes,iset=rhs_rids, darr=rhs_local) 

# load into distdata and multiply
rids = KK.getTree().getGIDS()
ww_rids = VecRedistribute(ww_user,comm,form='tree',tree_rids = rids) 
kde_density_rids = KK.evaluate(ww_rids)

# Store output in new object with old rids memberships
kde_density_user = VecRedistribute(kde_density_rids,comm,form='cyclic') 

"""
[6.] Compute errors, plots
"""

K1 = kde_density_user.toArray()
K1 -= rhs_local  # remove local contribution for self kernels
loc_class_mem_out = K1.argmax(axis=1)
loc_class_ids_out = class_list[loc_class_mem_out]
glob_acc = np.zeros(1)
loc_acc = np.count_nonzero(loc_class_ids_out == class_ids)* np.ones(1)
comm.Allreduce(loc_acc,glob_acc, op=MPI.SUM)
glob_acc /= n

print("Global KDE approximation accuracy at bw ",bandwidth," is ",glob_acc)



"""
[5.] Finalize environment
"""
prt.finalize()

