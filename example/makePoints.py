import numpy as np

N = 4000 #problem size
d = 2 #dimension

#two clusters of randomized points
class_1 = np.random.rand(d, (int)(np.floor(N/2)))*2
class_2 = np.random.rand(d, (int)(np.ceil(N/2)))*2 + 5

points = np.concatenate((class_1, class_2), axis=1) #shape = (d, N_per)
points = np.asarray(points, order='F', dtype='float32')

points.tofile("points")






