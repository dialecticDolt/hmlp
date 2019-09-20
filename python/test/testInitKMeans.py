import pygofmm.mltools.algorithms as alg
import numpy as np
from matplotlib import pyplot as plt

d = 2
n_local = 200

points = np.asarray(np.concatenate( (np.random.randn(d, n_local)-5, np.random.randn(d, n_local), np.random.randn(d, n_local)+10), axis=1), dtype='float32')
centers = alg.local_kmeans_pp(points, 3)
print(centers)


plt.scatter(points[0, :], points[1, :])
plt.scatter(centers[0, :], centers[1, :])
plt.show()
