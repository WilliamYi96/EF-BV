#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J fiEF21_mushrooms_o1
#SBATCH -o fiEF21_mushrooms_o1.%J.out
#SBATCH -e fiEF21_mushrooms_o1.%J.err
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
# n = 20, datasets = mushrooms, overlap = 1, Top1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 20  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 20  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1

# n = 20, datasets = mushrooms, overlap = 1, Top2
python compTRk.py --k 2 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 20  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1
python compTRk.py --k 2 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 20  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1

# n = 20, datasets = mushrooms, overlap = 1, MixK
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 20  --method EF21 --compressor MixK --ratio 1 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 20  --method EF21 --compressor MixK --ratio 1 --ind --overlap 1

# n = 20, datasets = mushrooms, overlap = 1, CompK (sqrt_d)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 20  --method EF21 --compressor CompK --ratio 10 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 20  --method EF21 --compressor CompK --ratio 10 --ind --overlap 1

# n = 20, datasets = mushrooms, overlap = 1, CompK (d/2)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 20  --method EF21 --compressor CompK --ratio 56 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 20  --method EF21 --compressor CompK --ratio 56 --ind --overlap 1

# ===============================================================
# ===============================================================
# n = 100, datasets = mushrooms, overlap = 1, Top1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 100  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 100  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1

# n = 100, datasets = mushrooms, overlap = 1, Top2
python compTRk.py --k 2 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 100  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1
python compTRk.py --k 2 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 100  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1

# n = 100, datasets = mushrooms, overlap = 1, MixK
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 100  --method EF21 --compressor MixK --ratio 1 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 100  --method EF21 --compressor MixK --ratio 1 --ind --overlap 1

# n = 100, datasets = mushrooms, overlap = 1, CompK (sqrt_d)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 100  --method EF21 --compressor CompK --ratio 10 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 100  --method EF21 --compressor CompK --ratio 10 --ind --overlap 1

# n = 100, datasets = mushrooms, overlap = 1, CompK (d/2)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 100  --method EF21 --compressor CompK --ratio 56 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 100  --method EF21 --compressor CompK --ratio 56 --ind --overlap 1

# ===============================================================
# ===============================================================
# n = 1000, datasets = mushrooms, overlap = 1, Top1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1

# n = 1000, datasets = mushrooms, overlap = 1, Top2
python compTRk.py --k 2 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1
python compTRk.py --k 2 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1

# n = 1000, datasets = mushrooms, overlap = 1, MixK
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF21 --compressor MixK --ratio 1 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF21 --compressor MixK --ratio 1 --ind --overlap 1

# n = 1000, datasets = mushrooms, overlap = 1, CompK (sqrt_d)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF21 --compressor CompK --ratio 10 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF21 --compressor CompK --ratio 10 --ind --overlap 1

# n = 1000, datasets = mushrooms, overlap = 1, CompK (d/2)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 1000  --method EF21 --compressor CompK --ratio 56 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 1000  --method EF21 --compressor CompK --ratio 56 --ind --overlap 1

# ===============================================================
# ===============================================================
# n = 8124, datasets = mushrooms, overlap = 1, Top1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 8124  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 8124  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1

# n = 8124, datasets = mushrooms, overlap = 1, Top2
python compTRk.py --k 2 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 8124  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1
python compTRk.py --k 2 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 8124  --method EF21 --compressor TopK --ratio 1 --ind --overlap 1

# n = 8124, datasets = mushrooms, overlap = 1, MixK
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 8124  --method EF21 --compressor MixK --ratio 1 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 8124  --method EF21 --compressor MixK --ratio 1 --ind --overlap 1

# n = 8124, datasets = mushrooms, overlap = 1, CompK (sqrt_d)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 8124  --method EF21 --compressor CompK --ratio 10 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 8124  --method EF21 --compressor CompK --ratio 10 --ind --overlap 1

# n = 8124, datasets = mushrooms, overlap = 1, CompK (d/2)
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 1 --num_workers 8124  --method EF21 --compressor CompK --ratio 56 --ind --overlap 1
python compTRk.py --k 1 --dataset mushrooms --max_it 100000 --tol 1e-7 --factor 8 --num_workers 8124  --method EF21 --compressor CompK --ratio 56 --ind --overlap 1