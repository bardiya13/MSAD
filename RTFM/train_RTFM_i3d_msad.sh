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
python3 main.py --feat-extractor 'i3d' --feature-size 2048 --rgb-list 'list/msad-i3d.list' \
        --test-rgb-list 'list/msad-i3d-test.list' --gt default='/scratch/kf09/lz1278/MSAD-Swin-WS/gt-MSAD-WS-new.npy' \
        --dataset 'msad' 2>&1 | tee ./train_logs_i3d_msad.txt
           
