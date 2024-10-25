#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16384MB
#PBS -l walltime=00:10:00
#PBS -l wd
#PBS -l storage=scratch/kf09


cd /scratch/kf09/lz1278/RTFM
module load python3/3.10.0
# python3 test.py 2>&1 --test-rgb-list 'list/shanghai-i3d-test-10crop.list' \
#     --gt 'list/gt-sh2.npy'   \
#     --testing-model 'ckpt/msad-i3d0.8786670186595432-2465.pkl' \
#     --dataset 'shanghai' \
#     | tee ./test_logs_msad_to_sht.txt

# python3 test.py 2>&1 --test-rgb-list 'list/ped2-i3d.list' \
#     --gt 'list/gt-ped2.npy'   \
#     --testing-model 'ckpt/msad-i3d0.8786670186595432-2465.pkl' \
#     --dataset 'ped2' \
#     | tee ./test_logs_msad_to_ped2.txt


# python3 test.py 2>&1 --test-rgb-list 'list/ped2-i3d-train.list' \
#     --gt 'list/gt-ped2.npy'   \
#     --testing-model 'ckpt/msad-i3d0.8786670186595432-2465.pkl' \
#     --dataset 'ped2' \
#     | tee ./test_logs_msad_to_ped2.txt


python3 test.py 2>&1 --test-rgb-list 'list/cuhk-i3d-test.list' \
    --gt '/scratch/kf09/lz1278/cuhk_i3d_feature/gt-cuhk.npy'   \
    --testing-model 'ckpt/ucf-i3d0.8346095811174203-1080.pkl' \
    --dataset 'cuhk' \
    | tee ./test_logs_ucf_to_cuhk.txt


