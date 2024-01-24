#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J EF22_CompK_phishing
#SBATCH -o EF22_CompK_phishing.%J.out
#SBATCH -e EF22_CompK_phishing.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

# shellcheck disable=SC2164
cd /ibex/ai/home/yik/EF22/EF22/Least_squares_PL-setting/
#cd /ibex/scratch/yik/EF22/EF22/Least_squares_PL-setting/
# EF22, phishing, CompK

# EF22 phishing, CompK
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 1 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 2 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 4 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 8 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 16 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 32 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind

python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 64 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 128 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 256 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 512 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind
python compTRk.py --k 1 --dataset phishing --max_it 10000 --tol 1e-7 --factor 1024 --num_workers 20  --method EF22 --compressor CompK --ratio 1 --ind








