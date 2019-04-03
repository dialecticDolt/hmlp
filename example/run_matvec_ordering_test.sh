mpirun -n 1 ./my_gofmm_test
cat temp.data_r0.bin > temp.data_p1
cat temp.gids_r0.bin > temp.gids_p1
mpirun -n 2 ./my_gofmm_test
cat temp.data_r0.bin temp.data_r1.bin > temp.data_p2
cat temp.gids_r0.bin temp.gids_r1.bin > temp.gids_p2
mpirun -n 4 ./my_gofmm_test
cat temp.data_r0.bin temp.data_r1.bin temp.data_r2.bin temp.data_r3.bin > temp.data_p4
cat temp.gids_r0.bin temp.gids_r1.bin temp.gids_r2.bin temp.gids_r3.bin > temp.gids_p4

python analyze_matvec_ordering.py


