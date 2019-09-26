#!/bin/bash

echo "-------Starting Sphere Test-----------"

d_list=(3)
bandwidth_list=(0.04 0.05 0.07 0.1 0.3 0.5)
budget_list=(0 0.01 0.05 0.1 0.5)
leaf_list=(256 512)
neighbor_list=(128 256)
secure_list=(0 1)
rank_list=(1 2)

#bandwidth_list=(0.05)
#budget_list=(0 0.1)
#leaf_list=(256)
#neighbor_list=(128)

export OMP_NUM_THREADS=10

i=0
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
                            echo "... Launching Job for Bandwidth: $h; Budget: $b; Leaf Size: $l; kNN: $k; Secure $sec"
                            echo "... This is the $i th job launched (Repeat $R)"
                            python_output=$(timeout -k 1 600 python ClusterSphere.py -nclasses 3 -d $d -N 10000 -bandwidth $h -budget $b -leaf $l -k $k -secure $sec -max_rank $l)
                            NMI=$(echo $python_output | grep -oP '(?:NMI1)..\d*\.?\d*' |  grep -oP '\d+\.\d*')
                            pNMI=$(echo $python_output | grep -oP '(?:NMI2)..\d*\.?\d*' |  grep -oP '\d+\.\d*')
                            ARI=$(echo $python_output | grep -oP '(?:ARI)..\d*\.?\d*' |  grep -oP '\d+\.\d*')
                            timing=$(echo $python_output | grep -oP '(?:Time)..\d*\.?\d*' |  grep -oP '\d+\.\d*')
                            nan=$(echo $python_output | grep -oP 'nan' | head -n 1)
                            echo "NMI: $NMI"
                            echo "pNMI: $pNMI"
                            echo "ARI: $ARI"
                            echo "timing: $timing"
                            echo "nan: $nan"
                            $(echo "$i, $d, $h, $b, $l, $k, $sec, $pNMI, $ARI, $timing, $nan" >> secure_output.csv)
                        done
                    done
                done
            done
        done
    done
done
