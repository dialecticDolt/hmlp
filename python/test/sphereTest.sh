#!/bin/bash

echo "-------Starting Sphere Test-----------"

d_list=(3)
bandwidth_list=(0.04 0.05 0.07 0.1 0.3 0.5)
budget_list=(0 0.01 0.05 0.1 0.5)
leaf_list=(128 256 512)
neighbor_list=(64 128 256)
secure_list=(0 1)
tol_list=(0.1 0.01 0.0001)
slack_list=(0, 1, -1, 2)
#bandwidth_list=(0.05)
#budget_list=(0 0.1)
#leaf_list=(256)
#neighbor_list=(128)

export OMP_NUM_THREADS=10

i=0
for t in "${tol_list[@]}"
do
    for d in "${d_list[@]}"
    do
        for h in "${bandwidth_list[@]}"
        do
            for b in "${budget_list[@]}"
            do
                for l in "${leaf_list[@]}"
                do
                    for k in "${neighbor_list[@]}"
                    do
                        for sec in "${secure_list[@]}"
                        do
                            for R in 1 2 3
                            do
                                i=$(($i+1))
                                echo "... Launching Job for Bandwidth: $h; Budget: $b; Leaf Size: $l; kNN: $k; Secure $sec, Tol: $t"
                                echo "... This is the $i th job launched (Repeat $R)"
                                python_output=$(timeout -k 1 600 python ClusterSphere.py -nclasses 3 -d $d -N 10000 -bandwidth $h -budget $b -leaf $l -k $k -secure $sec -max_rank $l -tolerance $t -slack 1)
                                NMI=$(echo $python_output | grep -oP '(?:NMI)..\d*\.?\d*(e)?(-)?\d*' |  grep -oP '\d+\.\d*(e)?(-)?\d*')
                                ARI=$(echo $python_output | grep -oP '(?:ARI)..\d*\.?\d*(e)?(-)?\d*' |  grep -oP '\d+\.\d*(e)?(-)?\d*')
                                timing=$(echo $python_output | grep -oP '(?:Total Time)..\d*\.?\d*(e)?(-)?\d*' |  grep -oP '\d+\.\d*(e)?(-)?\d*')
                                nan=$(echo $python_output | grep -oP 'nan' | head -n 1)
                                compress=$(echo $python_output | grep -oP '\d*\.?\d*(e)?(-)?\d*% uncompressed' | grep -oP '\d+\.\d*(e)?(-)?\d*')
                                estime=$(echo $python_output | grep -oP '(?:Eigensolver Time)..\d*\.?\d*(e)?(-)?\d*' |  grep -oP '\d+\.\d*(e)?(-)?\d*')

                                echo "NMI: $NMI"
                                echo "ARI: $ARI"
                                echo "Time: $timing"
                                echo "ES Time: $estime"
                                echo "nan: $nan"
                                echo "compress: $compress"
                                $(echo "$i, $d, $h, $b, $l, $k, $tol, $sec, $pNMI, $ARI, $timing, $estime, $nan, $compress" >> 10_1_3class_slack1_sphere_output.csv)
                            done
                        done
                    done
                done
            done
        done
    done
done
