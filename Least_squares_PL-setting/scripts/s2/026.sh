#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J fiEF22_a9a_o10
#SBATCH -o fiEF22_a9a_o10.%J.out
#SBATCH -e fiEF22_a9a_o10.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# shellcheck disable=SC2164
cd /ibex/ai/home/yik/EF22/EF22/Logistic_regression/
# EF22, a9a, CompK

# ===============================================================
# ===============================================================
# n = 1000, datasets = a9a, overlap = 1, MixK
python compTRk.py --k 1 --dataset a9a --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10
python compTRk.py --k 1 --dataset a9a --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10

# n = 1000, datasets = a9a, overlap = 1, CompK (sqrt_d)
python compTRk.py --k 1 --dataset a9a --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF22 --compressor CompK --ratio 25 --ind --overlap 10
python compTRk.py --k 1 --dataset a9a --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF22 --compressor CompK --ratio 25 --ind --overlap 10

# n = 1000, datasets = a9a, overlap = 1, CompK (d/2)
python compTRk.py --k 1 --dataset a9a --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF22 --compressor CompK --ratio 150 --ind --overlap 10
python compTRk.py --k 1 --dataset a9a --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF22 --compressor CompK --ratio 150 --ind --overlap 10