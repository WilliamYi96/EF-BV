#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J fiEF22_mushrooms_o10
#SBATCH -o fiEF22_mushrooms_o10.%J.out
#SBATCH -e fiEF22_mushrooms_o10.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# shellcheck disable=SC2164
cd /ibex/ai/home/yik/EF22/EF22/Logistic_regression/
# EF22, mushrooms, CompK
  
# ===============================================================
# ===============================================================
# n = 20, datasets = mushrooms, overlap = 1, MixK 
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 20  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 20  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10

# n = 20, datasets = mushrooms, overlap = 1, CompK (sqrt_d)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 20  --method EF22 --compressor CompK --ratio 10 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 20  --method EF22 --compressor CompK --ratio 10 --ind --overlap 10

# n = 20, datasets = mushrooms, overlap = 1, CompK (d/2)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 20  --method EF22 --compressor CompK --ratio 56 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 20  --method EF22 --compressor CompK --ratio 56 --ind --overlap 10

# ===============================================================
# ===============================================================
# n = 100, datasets = mushrooms, overlap = 1, MixK
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 100  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 100  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10

# n = 100, datasets = mushrooms, overlap = 1, CompK (sqrt_d)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 100  --method EF22 --compressor CompK --ratio 10 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 100  --method EF22 --compressor CompK --ratio 10 --ind --overlap 10

# n = 100, datasets = mushrooms, overlap = 1, CompK (d/2)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 100  --method EF22 --compressor CompK --ratio 56 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 100  --method EF22 --compressor CompK --ratio 56 --ind --overlap 10

# ===============================================================
# ===============================================================
# n = 1000, datasets = mushrooms, overlap = 1, MixK
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10

# n = 1000, datasets = mushrooms, overlap = 1, CompK (sqrt_d)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF22 --compressor CompK --ratio 10 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF22 --compressor CompK --ratio 10 --ind --overlap 10

# n = 1000, datasets = mushrooms, overlap = 1, CompK (d/2)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF22 --compressor CompK --ratio 56 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF22 --compressor CompK --ratio 56 --ind --overlap 10

# ===============================================================
# ===============================================================
# n = 8124, datasets = mushrooms, overlap = 1, MixK
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 8124  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 8124  --method EF22 --compressor MixK --ratio 1 --ind --overlap 10

# n = 8124, datasets = mushrooms, overlap = 1, CompK (sqrt_d)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 8124  --method EF22 --compressor CompK --ratio 10 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 8124  --method EF22 --compressor CompK --ratio 10 --ind --overlap 10

# n = 8124, datasets = mushrooms, overlap = 1, CompK (d/2)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 8124  --method EF22 --compressor CompK --ratio 56 --ind --overlap 10
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 8124  --method EF22 --compressor CompK --ratio 56 --ind --overlap 10