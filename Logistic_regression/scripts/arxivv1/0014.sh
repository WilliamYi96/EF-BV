#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J 0014
#SBATCH -o 0014.%J.out
#SBATCH -e 0014.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# shellcheck disable=SC2164
cd /ibex/ai/home/yik/EF22/EF22/Logistic_regression/
# EF22, mushrooms, CompK

python compTRk.py --k 2 --dataset a9a --max_it 100000 --tol 1e-7 --factor 1 --num_workers 32561 --method EF21 --compressor CompK --ratio 61 --ind --overlap 1

