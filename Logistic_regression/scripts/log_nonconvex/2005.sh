#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J 888
#SBATCH -o 888.%J.out
#SBATCH -e 888.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# shellcheck disable=SC2164
cd /ibex/ai/home/yik/EF22/EF22/Logistic_regression/
# EF22, mushrooms, CompK

# with smaller n_workers = 20
#python compTRk2.py --k 1 --dataset mushrooms --max_it 50000 --tol 1e-7 --factor 1 --num_workers 20 --method EF21 --compressor CompK --ratio 56 --ind --overlap 1
#python compTRk2.py --k 1 --dataset mushrooms --max_it 50000 --tol 1e-7 --factor 1 --num_workers 20 --method EF22 --compressor CompK --ratio 56 --ind --overlap 1

#python compTRk2.py --k 1 --dataset w8a --max_it 50000 --tol 1e-7 --factor 1 --num_workers 20 --method EF21 --compressor CompK --ratio 150 --ind --overlap 1
#python compTRk2.py --k 1 --dataset w8a --max_it 50000 --tol 1e-7 --factor 1 --num_workers 20 --method EF22 --compressor CompK --ratio 150 --ind --overlap 1
#
#python compTRk2.py --k 1 --dataset phishing --max_it 50000 --tol 1e-7 --factor 1 --num_workers 20 --method EF21 --compressor CompK --ratio 34 --ind --overlap 1
#python compTRk2.py --k 1 --dataset phishing --max_it 50000 --tol 1e-7 --factor 1 --num_workers 20 --method EF22 --compressor CompK --ratio 34 --ind --overlap 1
#
#python compTRk2.py --k 1 --dataset a9a --max_it 50000 --tol 1e-7 --factor 1 --num_workers 20 --method EF21 --compressor CompK --ratio 61 --ind --overlap 1
#python compTRk2.py --k 1 --dataset a9a --max_it 50000 --tol 1e-7 --factor 1 --num_workers 20 --method EF22 --compressor CompK --ratio 61 --ind --overlap 1
#
## n = 50
python compTRk2.py --k 1 --dataset mushrooms --max_it 50000 --tol 1e-7 --factor 1 --num_workers 50 --method EF21 --compressor CompK --ratio 56 --ind --overlap 1
python compTRk2.py --k 1 --dataset mushrooms --max_it 50000 --tol 1e-7 --factor 1 --num_workers 50 --method EF22 --compressor CompK --ratio 56 --ind --overlap 1
#
#python compTRk2.py --k 1 --dataset w8a --max_it 50000 --tol 1e-7 --factor 1 --num_workers 50 --method EF21 --compressor CompK --ratio 150 --ind --overlap 1
#python compTRk2.py --k 1 --dataset w8a --max_it 50000 --tol 1e-7 --factor 1 --num_workers 50 --method EF22 --compressor CompK --ratio 150 --ind --overlap 1
#
#python compTRk2.py --k 1 --dataset phishing --max_it 50000 --tol 1e-7 --factor 1 --num_workers 50 --method EF21 --compressor CompK --ratio 34 --ind --overlap 1
#python compTRk2.py --k 1 --dataset phishing --max_it 50000 --tol 1e-7 --factor 1 --num_workers 50 --method EF22 --compressor CompK --ratio 34 --ind --overlap 1
#
#python compTRk2.py --k 1 --dataset a9a --max_it 50000 --tol 1e-7 --factor 1 --num_workers 50 --method EF21 --compressor CompK --ratio 61 --ind --overlap 1
#python compTRk2.py --k 1 --dataset a9a --max_it 50000 --tol 1e-7 --factor 1 --num_workers 50 --method EF22 --compressor CompK --ratio 61 --ind --overlap 1


