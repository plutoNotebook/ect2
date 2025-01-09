#!/usr/bin/bash

#SBATCH -J ecd-stf2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 6-0
#SBATCH -o logs/slurm-%A.out

bash run_ecm.sh 2 6010 --desc base.stf2 --ecd=True --outdir=ecd

exit 0
