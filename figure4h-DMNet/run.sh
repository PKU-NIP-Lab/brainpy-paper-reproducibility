#!/bin/bash
#SBATCH -J std1
#SBATCH -o std1.out
#SBATCH -p q_ai
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 30G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 1

python brainpy_implementation.py gpu