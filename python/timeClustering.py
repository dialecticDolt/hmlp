import pygofmm.core as PyGOFMM
import pygofmm.mltools.algorithms as alg

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

rt = PyGOFMM.PyRuntime()
rt.init_with_MPI(comm)

N = 10000*nprocs;
d = 5


#Construct the point set
np.random.seed(10)
class_1 = np.random.randn(d, (int)(np.floor(N/3))) + 5
class_2 = np.random.randn(d, (int)(np.floor(N/3)))
class_3 = np.random.randn(d, (int)(np.ceil(N/3)))
test_points = np.asarray(np.concatenate((class_1, class_2, class_3), axis=1), dtype='float32')

c1 = np.ones(np.shape(class_1))
c2 = np.ones(np.shape(class_2))+1
c3 = np.ones(np.shape(class_3))+1

true_classes = np.asarray(np.concatenate((c1, c2, c3), axis=1), dtype='float32', order='F')

#Redistribute points to cyclic partition
start_s = MPI.Wtime()
sources, GIDS_Owned = PyGOFMM.CBLK_Distribute(comm, test_points)

true_classes = np.asarray(true_classes[0, GIDS_Owned], dtype='int32').flatten()

#Median Trick. Estimate scale of bandwidth

k = 64
PyNNList = PyGOFMM.FindAllNeighbors(comm, N, k, sources)
distances, gids = PyNNList.toNumpy()
a = np.median(distances[1:10, 1:])
b = np.median(distances[:, 1:])

a = 0.2
a = 5
bandwidths = np.logspace(np.log2(a), np.log2(b), num=4, base=2)

spec = True;
if spec:
        t = "spectral"
else:
        t = "kernel"

L = 128 #leafnodesize
k = 64 #number of neighbors
tol = 1E-5
budget = 0

results = open(t+"_clustering_tol"+str(tol)+"_budget"+str(budget)+"_MPI_"+str(nprocs)+".txt", "a")


for lam in bandwidths:
        #Setup and compress the kernel matrix
        config = PyGOFMM.PyConfig("GEOMETRY_DISTANCE", N, L, k, L, tol, budget, False)
        K = PyGOFMM.KernelMatrix(comm, sources, conf=config, bandwidth=lam)

        end_s = MPI.Wtime()
        start_co = MPI.Wtime()
        K.compress()
        end_co = MPI.Wtime()


        #Run Clustering
        nclasses = 2
        start_c = MPI.Wtime()
        if spec:
            classes, eig_time, center_time, update_time, init_time  = alg.SpecCluster(K, nclasses, gids=GIDS_Owned)
        else:
            classes, matvec_time, np_time, sim_time, com_time = PyGOFMM.FastKKMeans(K,  nclasses, maxiter=20, gids=GIDS_Owned)
        end_c = MPI.Wtime()

        classes = np.asarray(classes, dtype='int32').flatten()
        
        if rank == 0:
                results.write("\nB: "+str(lam)+"\n")
                results.write("NMI: "+str(alg.NMI(comm, true_classes, classes, nclasses))+"\n")
                results.write("Setup Time: " + str(end_s - start_s)+"\n")
                results.write("Compress Time: "+str(end_co - start_co)+"\n")
                results.write("Total Clustering Time: "+ str(end_c - start_c)+"\n")
                if spec:
                        results.write("KMeans++: "+str(init_time)+"\n")
                        results.write("KMeans - Update Means:"+str(center_time)+"\n")
                        results.write("KMeans - Compute Classes: "+str(update_time)+"\n")
                        results.write("Eigenvector: "+str(-1*eig_time)+"\n")
                else:
                        results.write("Total Matvec Time: "+str(matvec_time)+"\n")
                        results.write("Lookup Matricies: "+str(np_time)+"\n")
                        results.write("Update Classes: "+str(sim_time)+"\n")
                        results.write("Communication Time: "+str(com_time)+"\n")
                                
        print("Bandwidth", lam)
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

results.close()
rt.finalize()
