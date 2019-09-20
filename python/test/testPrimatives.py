import pygofmm.mltools.algorithms as alg
import numpy as np


a = np.asarray([0, 0.1, 0.2, 0.8 , 0.9], dtype='float32')

ind = alg.local_bsearch(0.09, a)
print(a)
print(ind)
