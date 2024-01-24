#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J ef21_dl1
#SBATCH -o ef21_dl1.%J.out
#SBATCH -e ef21_dl1.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=144:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:4

# shellcheck disable=SC2164
cd /ibex/ai/home/yik/EF22/EF21/DL/

python EF21_100K.py --factor 8 --max_it 4545 --k 1320000 --batch_size 2048 --model vgg11 --dataset CIFAR10

