import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from numpy.linalg import norm
import itertools
from scipy.stats import ortho_group
import pandas as pd
from matplotlib import pyplot as plt
import math

import datetime
from IPython import display
from least_squares_functions_fast import *

from contextlib import redirect_stdout
import shutil
import subprocess

# test for single plot only
def plot1(x_ar, y_ar, label_ar, plot_path, dataset, title=None, xaxis='bits/n', yaxis=r"$\|| \nabla f(x) \||^2$",
          yscale="log", xscale="non-log", filename=None, save=0):
    size = 20
    marker_size = 20
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'FreeSerif'
    plt.rcParams['lines.linewidth'] = 4
    # plt.rcParams['lines.markersize'] = 10
    plt.rcParams['xtick.labelsize'] = size  # 40
    plt.rcParams['ytick.labelsize'] = size  # 40
    plt.rcParams['legend.fontsize'] = size  # 30
    plt.rcParams['axes.titlesize'] = size  # 40
    plt.rcParams['axes.labelsize'] = size  # 40
    plt.rcParams["figure.figsize"] = [13, 9]
    if yscale == "log":
        plt.yscale('log')
    if xscale == "log":
        plt.xscale('log')
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    color_ar_1 = ['blue', 'red', 'orange', 'coral', 'violet', 'brown'] + ['darkorange', 'cornflowerblue', 'darkgreen',
                                                                'coral', 'lime',
                                                                'darkgreen', 'goldenrod', 'maroon',
                                                                'black', 'brown', 'yellowgreen', "purple", "violet",
                                                                "magenta", "green"
                                                                ]
    marker_ar = ["o", "*", "v", "^", "<", ">", "s", "p", "P", "h", "H", "+", "x", "X", "D", "d", "|", "_", 1, 2, 3, 4,
                 5, 6, 7, 8, 9]
    if title is None:
        title = f"Logistic regression with non-convex regularizer; {dataset}"
    plt.title(title)

    # print (len(x_ar), len(marker_ar), len(color_ar_1))  # (6, 27, 21)

    for i in range(len(x_ar)):
        # print (x_ar[i].shape[0], y_ar[i].shape[0]) # (10002), (10002)
        inds = np.arange(x_ar[i].shape[0])
        markers_on = inds[inds % (int(len(inds[:-(1 + 2 * i)]) / 10)) == 0].astype(int)

        plt.plot(x_ar[i], y_ar[i], 'r', label=label_ar[i], color=color_ar_1[i], marker=marker_ar[i],
                 markevery=list(markers_on), markersize=marker_size)
    # plt.plot(its_rand_k[i], norms_rand_k[i], 'r', animated=True, label=f'Rand-k; k={k_ar[i]}', color=color_ar_2[i])

    legend = plt.legend(loc="upper right", framealpha=0.5)
    # plt.legend()
    if save:
        plt.savefig(plot_path + filename, bbox_inches='tight')
    plt.show()


project_path = os.getcwd() + "/"
# dataset = 'mushrooms'
# dataset = 'w8a'
# dataset = 'a9a' #58753
# dataset = 'mushrooms'  # 58753
dataset = 'phishing' #58753

data_path = project_path + "data_{0}/".format(dataset)
plot_path = project_path + "plot_{0}/".format(dataset)

# k_size_ar_bd= [32]
k_size_ar_bd_ar = [[1], [2], [4], [8], [16], [32], [64]]

# k_size_ar_bd_ar = [[112]]
k_size_ar_bd_ar = [[1]]
# k_size_ar_bd_ar = [[2]]

# k_size_ar_bd_ar = [[64],[128],[256]]
# k_size_ar_bd_ar = [[8]]
for k_size_ar_bd in k_size_ar_bd_ar:
    # for k_size_ar_bd in k_size_ar_bd_ar:
    # k_size_ar_bd= [1,2,4,8,16,32,64]

    n_ar = np.array([20], dtype=int)

    # main_title = "comparison_best"
    # main_title = "ef_evo"
    # main_title = "bd_evo"
    # main_title = "20_workers_300K_rand_k"
    main_title = f"EF21_{dataset}_20workers_300K_Top{k_size_ar_bd[0]}K"

    # factor_ar = [4,8]
    # factor_ar_bd = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    # factor_ar_bd = [1, 2, 4, 8, 16, 32]
    # factor_ar_bd = [8]
    factor_ar_bd = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # factor_ar_bd = [64]
    # factor_ar_bd = [1,2,4,8,16,32]
    # factor_ar_bd = [1]

    its_ar = []
    norms_ar = []
    # label_ar = []

    ub_bits = 20_100_000

    experiment_ar = ["EF21-full-grad_nw-{0}_{1}x_TopK_{2}".format(n, factor_bd, k) for n, k, factor_bd in
                     itertools.product(n_ar, k_size_ar_bd, factor_ar_bd)]
    label_ar = ["EF22; k={0}; {1}x".format(k_size, factor_bd) for k_size, factor_bd in
                itertools.product(k_size_ar_bd, factor_ar_bd)]

    for i, experiment in enumerate(experiment_ar):
        logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)
        its = np.load(logs_path + 'iteration' + '_' + experiment + '.npy')
        number_its = len(its[its < ub_bits])
        its_ar.append(its[:number_its])  # (10002, 6)

        norms = np.load(logs_path + 'norms' + '_' + experiment + '.npy')
        norms_ar.append(norms[:number_its])

        print(experiment, f"iter: {its_ar[-1].shape[0]};", f"bits/n: {its_ar[-1][-1]}")

    filename = "{0}_{1}.pdf".format(main_title, dataset)
    use_top_iter = 4
    plot1(its_ar, norms_ar, label_ar, plot_path, dataset, filename=filename, save=1)