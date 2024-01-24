#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J EF21_CompK_mushrooms_2
#SBATCH -o EF21_CompK_mushrooms_2.%J.out
#SBATCH -e EF21_CompK_mushrooms_2.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# shellcheck disable=SC2164
cd /ibex/ai/home/yik/EF22/EF22/Least_squares_PL-setting/
#cd /ibex/scratch/yik/EF22/EF22/Least_squares_PL-setting/
# EF21, mushrooms, CompK

# EF21 mushrooms, CompK
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 2 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 4 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 8 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 16 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 32 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 64 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 128 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 256 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 512 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind
python compTRk.py --k 1 --dataset mushrooms --max_it 10000 --tol 1e-7 --factor 1024 --num_workers 20  --method EF21 --compressor CompK --ratio 2 --ind








