mpirun -n 1 ./readPoints
cat temp.data_r0 > temp.data_p1
cat temp.gids_r0 > temp.gids_p1
mpirun -n 2 ./readPoints
cat temp.data_r0 temp.data_r1 > temp.data_p2
cat temp.gids_r0 temp.gids_r1 > temp.gids_p2
mpirun -n 4 ./readPoints
cat temp.data_r0 temp.data_r1 temp.data_r2 temp.data_r3 > temp.data_p4
cat temp.gids_r0 temp.gids_r1 temp.gids_r2 temp.gids_r3 > temp.gids_p4

python analyze_matvec_ordering.py


