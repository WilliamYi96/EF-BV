#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J 0013
#SBATCH -o 0013.%J.out
#SBATCH -e 0013.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# shellcheck disable=SC2164
cd /ibex/ai/home/yik/EF22/EF22/Logistic_regression/
# EF22, mushrooms, CompK

python compTRk.py --k 2 --dataset phishing --max_it 100000 --tol 1e-7 --factor 1 --num_workers 11055 --method EF21 --compressor CompK --ratio 34 --ind --overlap 1

