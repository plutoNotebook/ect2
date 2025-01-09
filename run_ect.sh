#!/usr/bin/bash

#SBATCH -J ecm-run
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 6-0
#SBATCH -o logs/slurm-%A.out

bash run_ecm.sh 2 6007 --desc bs128.200k --outdir=vanilla

exit 0
