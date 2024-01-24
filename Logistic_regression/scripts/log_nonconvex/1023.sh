#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J 1023
#SBATCH -o 1023.%J.out
#SBATCH -e 1023.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# shellcheck disable=SC2164
cd /ibex/ai/home/yik/EF22/EF22/Logistic_regression/
# EF22, mushrooms, CompK

python compTRk2.py --k 1 --dataset phishing --max_it 100000 --tol 1e-7 --factor 1 --num_workers 11055 --method EF21 --compressor CompK --ratio 34 --ind --overlap 2

