import pygofmm.core as PyGOFMM
import pygofmm.mltools.algorithms as alg

from mpi4py import MPI
import numpy as np

#import matplotlib.pyplot as plt
#import plotly
#plotly.tools.set_credentials_file(username='wlruys', api_key='mzrS27JzVlqXnVLBsJnB')
#import plotly.plotly as py
#import plotly.graph_objs as go

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

rt = PyGOFMM.PyRuntime()
rt.init_with_MPI(comm)

N = 100000*nprocs+8
d = 5


#Construct the point set
np.random.seed(10)
class_1 = np.random.randn(d, (int)(np.floor(N/3))) + 4
class_2 = np.random.randn(d, (int)(np.floor(N/3)))
class_3 = np.random.randn(d, (int)(np.ceil(N/3)))
test_points = np.asarray(np.concatenate((class_1, class_2, class_3), axis=1), dtype='float32')
#print(np.shape(test_points))
c1 = np.ones(np.shape(class_1))
c2 = np.ones(np.shape(class_2))+1
c3 = np.ones(np.shape(class_3))+1

true_classes = np.asarray(np.concatenate((c1, c2, c3), axis=1), dtype='float32', order='F')

#t = np.linspace(0, 10*3.1415, N/2)
#noise = np.random.randn(1, int(N/2))*0.1
#class_1 = [np.sin(t)*5, np.cos(t)*5, t]
#class_2 = [np.sin(t), np.cos(t), t]
#test_points = np.concatenate((class_1, class_2), axis=1)
#
#for i in range(N):
#    if i < N/2:
#        test_points[:, i] = [np.sin(3*t[i%(N/2)])*5, np.cos(3*t[i%(N/2)])*5, t[3*i%(N/2)]]
#    else:
#        test_points[:, i] = [np.sin(8*t[i%(N/2)]), np.cos(8*t[i%(N/2)]), t[8*i%(N/2)]]
#
#print(np.shape(test_points))

#test_points = np.asarray(test_points.T, dtype='float32')
#c1 = np.ones(np.shape(class_1))
#c2 = np.ones(np.shape(class_2))+1
#true_classes = np.asarray(np.concatenate((c1, c2)), dtype='float32', order='F')

#Redistribute points to cyclic partition
start_s = MPI.Wtime()
sources, GIDS_Owned = PyGOFMM.CBLK_Distribute(comm, test_points)
#print(len(GIDS_Owned))

#Setup and compress the kernel matrix
config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, 128, 64, 128, 0.0001, 0, False)
K = PyGOFMM.KernelMatrix(comm, sources, conf=config, bandwidth=1.3)
end_s = MPI.Wtime()

start_co = MPI.Wtime()
K.compress()
end_co = MPI.Wtime()

spec = False

#Run kernel k-means
nclasses = 2
start_c = MPI.Wtime()
if spec:
    classes, eig_time, center_time, update_time, init_time  = alg.SpecCluster(K, nclasses, gids=GIDS_Owned)
else:
    classes, matvec_time, np_time, sim_time, com_time = PyGOFMM.FastKKMeans(K,  nclasses, maxiter=20, gids=GIDS_Owned)
end_c = MPI.Wtime()

#classes = alg.KKMeans(K, test_points, 3, maxiter=40, gids=GIDS_Owned)
#Redistribute points to original partition

#Gather points to root (for plotting)
#recvbuf = comm.allgather(classes)
#classes = np.concatenate(recvbuf, axis=0).astype('int32')

#recvbuf = comm.allgather(GIDS_Owned)
#gids = np.concatenate(recvbuf, axis=0).astype('int32')
#
#points = test_points[:, gids].T

true_classes = np.asarray(true_classes[0, GIDS_Owned], dtype='int32').flatten()
classes = np.asarray(classes, dtype='int32').flatten()

print(alg.NMI(comm, true_classes, classes, nclasses))
print("Setup: ", end_s - start_s)
print("Compress: ", end_co - start_co)
print("Clustering: ", end_c - start_c)

if spec:
    print("KMeans++: ", init_time)
    print("KMeans- Centroids: ", center_time)
    print("KMeans- Update: ", update_time)
    print("Eigenvectors: ", -1*eig_time)
else:
    print("MV", matvec_time)
    print("NP", np_time)
    print("SI", sim_time)
    print("com", com_time)

#plt.scatter(points[:, 0], points[:, 1], c=classes.flatten())
#plt.show()

#ones = np.ones([N, d], dtype='float32', order='F')
#test = PyGOFMM.PyDistData_RIDS(comm, N, d, darr=ones, iset=GIDS_Owned)
#alg.testDistDataArray(test)
#print(test.toArray())

#trace1 = go.Scatter3d(
#        x = points[:, 1],
#        y = points[:, 0],
#        z = points[:, 2],
#        mode = 'markers',
#        marker=dict(
#            size = 6,
#            opacity=0.8,
#            #line = dict(
#            #    color = 'rgb(256, 256, 256)',
#            #    width = 0.5
#            #    ),
#            color = classes.flatten(),
#            colorscale='Viridis'
#            )
#        )
#data = [trace1]
#py.iplot(data, filename='scatter-plot')
#

rt.finalize()
