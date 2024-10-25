#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16384MB
#PBS -l walltime=05:00:00
#PBS -l wd
#PBS -l storage=scratch/kf09


cd /scratch/kf09/lz1278/RTFM
module load python3/3.10.0
python3 main.py --feat-extractor 'i3d' --feature-size 2048 --rgb-list 'list/shanghai-i3d-train-10crop.list' \
        --test-rgb-list 'list/shanghai-i3d-test-10crop.list' --gt default='list/gt-sh2.npy' \
        --dataset 'shanghai' 2>&1 | tee ./train_logs_i3d_sht.txt
           
