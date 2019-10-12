#!/bin/bash
t=0.001
sec=0
d=5
c=2

export OMP_NUM_THREADS=6

n_list=(10000 20000 40000 80000)
bandwidth_list=(1)
leaf_list=(512 1024)
budget_list=(0 0.01 0.1 0.2)
neighbor_list=(256 10 5)
rank_list=(1024 2048)
secure_list=(0)
i=0
for l in "${leaf_list[@]}"
do
 for h in "${bandwidth_list[@]}"
 do
     for s in "${rank_list[@]}"
     do
        for b in "${budget_list[@]}"
        do
             for k in "${neighbor_list[@]}"
             do
                for n in "${n_list[@]}"
                do
                     output=$(timeout -k 1 400 python ClusterNonLinear.py -N $n -bandwidth $h -d $d -nclasses $c -leaf $l -max_rank $s -k $k -tolerance $t -budget $b -secure $sec -cluster sc -iter 10)
                     i=$(($i+1))
                     NMI=$(echo $output | grep -oP '(?:NMI)..\d*\.?\d*(e)?(-)?\d*' |  tail -n 1 | grep -oP '\d+\.?\d*(e)?(-)?\d*')
                     ARI=$(echo $output | grep -oP '(?:ARI)..\d*\.?\d*(e)?(-)?\d*' |  tail -n 1 | grep -oP '\d+\.?\d*(e)?(-)?\d*')
                     timing=$(echo $output | grep -oP '(?:Total Time)..\d*\.?\d*(e)?(-)?\d*' | tail -n 1 | grep -oP '\d+\.?\d*(e)?(-)?\d*')
                     comtiming=$(echo $output | grep -oP '(?:Compress Time)..\d*\.?\d*(e)?(-)?\d*' |  tail -n 1 | grep -oP '\d+\.?\d*(e)?(-)?\d*')
                     nan=$(echo $output | grep -oP 'nan' | head -n 1)

                     estime=$(echo $output | grep -oP '(?:Eigensolver Time)..\d*\.?\d*(e)?(-)?\d*' |  tail -n 1 |grep -oP '\d+\.?\d*(e)?(-)?\d*')
                     fnorm=$(echo $output | grep -oP '(?:GOFMM).\d*\.\d*(E)?(-)?\d*' | tail -n 1 | grep -oP '\d+\.?\d*(E)?(-)?\d*')

                     mtv=$(echo $output | grep -oP '(?:Evaluate..Async.).(-)*..\d*\.?\d*(e)?(-)?\d*' | tail -n 1 | grep -oP '\d+\.?\d*')

                     compress=$(echo $output | grep -oP '\d+\.\d*(e)?(-)?\d*% uncompressed' | grep -oP '\d+\.?\d*')
                     near_compress=$(echo $output | grep -oP '\d+\.\d*(e)?(-)?\d*% and' | grep -oP '\d+\.?\d*')

                     echo "NMI: $NMI"
                     echo "ARI: $ARI"
                     echo "Time: $timing"
                     echo "Com Time: $comtiming"
                     echo "ES Time: $estime"
                     echo "matvec time: $mtv"
                     echo "nan: $nan"
                     echo "far compress: $compress"
                     echo "near compress: $near_compress"
                     echo "FNORM: $fnorm"

                     $(echo "$n, $i, $d, $c, $h, $b, $l, $k, $t, $sec, $slack, $NMI, $ARI, $comtiming, $timing, $estime, $mtv, $near_compress, $compress, $fnorm, $s" >> quadratic_scaling_sc_nonlinear.csv)
                done
            done
        done
     done
 done
done
