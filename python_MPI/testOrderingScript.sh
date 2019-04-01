mpirun -n 1 python KKMeans.py
cat D_0 > D_1p

mpirun -n 2 python KKMeans.py
cat D_0 D_1 > D_2p

mpirun -n 4 python KKMeans.py
cat D_0 D_1 D_2 D_3 > D_4p

python testOutput.py
