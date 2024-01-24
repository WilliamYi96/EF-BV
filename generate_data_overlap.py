"""
A script for the data preprocessing before launching an algorithm
It takes the given dataset and outcomes the partition
"""

import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file, dump_svmlight_file
from numpy.linalg import norm
import itertools
from scipy.special import binom
from scipy.stats import ortho_group
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

parser = argparse.ArgumentParser(description='Generate data and provide information about it '
                                             'for workers and parameter server')
parser.add_argument('--dataset', type=str, default='mushrooms')
parser.add_argument('--root_path', type=str, default='/ibex/ai/home/yik/EF22/EF22/')
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--times', type=int, default=1, help='range from 1-nworkers')
parser.add_argument('--funcs', type=str, default='log_reg', help='choose from log_reg and least_square')

args = parser.parse_args()
nworkers, times = args.num_workers, args.times

cur_path = f'{args.root_path}data/data_{args.dataset}_{nworkers}'
out_path = f'{args.root_path}data/splits/data_{args.dataset}_{nworkers}_{times}'

print(out_path)
if not os.path.exists(out_path):
    os.mkdir(out_path)

# Process training data
for i in range(nworkers):
    if nworkers - i < times:
        cur_start = nworkers - times
    else:
        cur_start = i
    cur_end = cur_start + times - 1
    file_path = f'{cur_path}/X_{args.dataset}_nw{nworkers}_{cur_start}.npy'
    out = np.load(file_path)
    # make data sample numbers are the same
    if i == 0:
        unit_samples = out.shape[0]
    for j in range(cur_start+1, cur_end+1):
        file_path2 = f'{cur_path}/X_{args.dataset}_nw{nworkers}_{j}.npy'
        # print(i, file_path, file_path2)
        out2 = np.load(file_path2)[:unit_samples]
        out = np.vstack((out, out2))
    out_path_tmp = f'{out_path}/X_{args.dataset}_nw{nworkers}_{i}.npy'
    np.save(out_path_tmp, out)

# Process training label
for i in range(nworkers):
    if nworkers - i < times:
        cur_start = nworkers - times
    else:
        cur_start = i
    cur_end = cur_start + times - 1
    file_path = f'{cur_path}/y_{args.dataset}_nw{nworkers}_{cur_start}.npy'
    out = np.load(file_path)
    if i == 0:
        unit_samples = out.shape[0]
    for j in range(cur_start+1, cur_end+1):
        file_path2 = f'{cur_path}/y_{args.dataset}_nw{nworkers}_{j}.npy'
        # print(i, file_path, file_path2)
        out2 = np.load(file_path2)[:unit_samples]
        out = np.append(out, out2)
    out_path_tmp = f'{out_path}/y_{args.dataset}_nw{nworkers}_{i}.npy'
    np.save(out_path_tmp, out)

from shutil import copyfile
for i in range(1):
    init_org = f'{cur_path}/w_init_{i}_{args.funcs}.npy'
    init_out = f'{out_path}/w_init_{i}_{args.funcs}.npy'
    copyfile(init_org, init_out)

copyfile(f'{cur_path}/X.npy', f'{out_path}/X.npy')
copyfile(f'{cur_path}/y.npy', f'{out_path}/y.npy')
copyfile(f'{cur_path}/data_info.npy', f'{out_path}/data_info.npy')

