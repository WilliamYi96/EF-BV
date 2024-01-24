"""
Plot on the mushrooms dataset.
Created by Kai Yi on July 18th.
"""

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
from logreg_functions_fast import *

from contextlib import redirect_stdout
import shutil
import subprocess


def plot1(x_ar, y_ar, label_ar, plot_path, dataset, title=None, xaxis='bits/n', yaxis=r"$ f(x^t) - f^\star $",
          yscale="log", xscale="non-log", filename=None, save=0, nworkers=1000, k_size_compk=10, k_size_ar_bd=1, overlap=1, interval=0):
    size = 30
    size2 = 35
    marker_size = 20
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'FreeSerif'
    plt.rcParams['lines.linewidth'] = 4
    # plt.rcParams['lines.markersize'] = 10
    plt.rcParams['xtick.labelsize'] = 20  # 40
    plt.rcParams['ytick.labelsize'] = size  # 40
    plt.rcParams['legend.fontsize'] = size2  # 30
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
                                                                          'black', 'brown', 'yellowgreen', "purple",
                                                                          "violet",
                                                                          "magenta", "green"
                                                                          ]
    marker_ar = ["o", "*", "v", "^", "<", ">", "s", "p", "P", "h", "H", "+", "x", "X", "D", "d", "|", "_", 1, 2, 3, 4,
                 5, 6, 7, 8, 9]
    if title is None:
        # title = f"Logistic regression with convex regularizer; {dataset}, n={nworkers}, CompK, k'={k_size_compk}"
        title = fr"{dataset}, comp-({k_size_ar_bd},{k_size_compk}), $\xi$={overlap}"
    plt.title(title)

    for i in range(len(x_ar)):
        inds = np.arange(x_ar[i].shape[0])
        markers_on = inds[inds % (int(len(inds[:-(1 + 2 * i)]) / 10)) == 0].astype(int)

        label_ar[i] = label_ar[i].replace("k", "k'")
        label_ar[i] = label_ar[i].replace("EF22", "EF-BV")
        print(label_ar[i])

        plt.plot(x_ar[i][:interval], y_ar[i][:interval], 'r', label=label_ar[i], color=color_ar_1[i])

    legend = plt.legend(loc="lower left", framealpha=0.5)
    # plt.legend()
    if save:
        plt.savefig(plot_path + filename, bbox_inches='tight')
    plt.show()


import argparse
parser = argparse.ArgumentParser(description='Plot results')
parser.add_argument('--dataset', type=str, default='mushrooms', help='choose from a9a, w8a, mushrooms, phishing')
parser.add_argument('--kp', default=56, type=int, help="k' size")
parser.add_argument('--k', default=1, type=int, help='k size')
parser.add_argument('--lrf', default=1, type=int, help='learning rate scaling factor')
parser.add_argument('--ovf', default=1, type=int, help='distributed data overlap ratio')
parser.add_argument('--nworkers', default=20, type=int, help='number of distributed workers')
parser.add_argument('--setting', default='conv', type=str, help='choose from conv and nonconv')
args = parser.parse_args()

project_path1 = os.getcwd() + "/"
project_path2 = '/ibex/ai/home/yik/EF22/EF22/datalogs/'
dataset = args.dataset

data_path = project_path2 + "data_{0}/".format(dataset)
plot_path = project_path1 + "plots/plot_{0}/".format(dataset)

k_prime_size_ar = [args.kp]  # choose from 10 and 56
k_size_ar = [args.k]  # Perm-(k', k), k value arrays
lr_factor_ar = [args.lrf]  # learning rate multiply factor, >1 for stepsize tolerance checking
overlap_ar = [args.ovf]  # overlap ratio, choose from 1 and 2

for k_size_ar_bd in k_size_ar:

    n_ar = np.array([args.nworkers], dtype=int)
    main_title = f'CompK_{k_prime_size_ar[0]}_{k_size_ar_bd}_{dataset}_8124workers_3000K_{n_ar[0]}'

    its_ar = []
    norms_ar = []
    # label_ar = []

    ub_bits = 20_100_000

    experiment_ar = ["EF21-full-grad_nw-{0}_{1}x_CompK_{2}_True_{3}_{4}".format(n, lr_factor, k_prime, overlap, k) for n, lr_factor, k_prime, overlap, k in
                     itertools.product(n_ar, lr_factor_ar, k_prime_size_ar, overlap_ar, k_size_ar)] +\
                    ["EF22-full-grad_nw-{0}_{1}x_CompK_{2}_True_{3}_{4}".format(n, lr_factor, k_prime, overlap, k) for n, lr_factor, k_prime, overlap, k in
                     itertools.product(n_ar, lr_factor_ar, k_prime_size_ar, overlap_ar, k_size_ar)]
    label_ar = ["EF21"] + ["EF-BV"]

    fxStar = np.array([0.34426245])

    for i, experiment in enumerate(experiment_ar):
        logs_path = project_path2 + "logs_{0}_{1}/".format(dataset, experiment)
        its = np.load(logs_path + 'iteration' + '_' + experiment + '.npy')
        number_its = len(its[its < ub_bits])
        its_ar.append(its[:number_its])  # (10002, 6)

        norms_tmp = []
        norms = np.load(logs_path + 'diff_ar' + '_' + experiment + '.npy')
        for i in range(len(norms)):
            norms_tmp.append(np.sqrt(norms[i]))
        norms = np.squeeze(np.array([norms_tmp]))
        norms_ar.append(norms[:number_its])

        print(experiment, f"iter: {its_ar[-1].shape[0]};", f"bits/n: {its_ar[-1][-1]}")

    filename = f"{main_title}_{dataset}_{k_prime_size_ar[0]}_{k_size_ar[0]}_{lr_factor_ar[0]}_{overlap_ar[0]}_logreg_diff_{args.setting}.pdf"
    use_top_iter = 4
    interval = int(len(its_ar[0]) / 2) # mushrooms
    if dataset == 'a9a':
        interval = 50000
    elif dataset == 'w8a':
        interval = 80000
    elif dataset == 'phishing':
        interval = 4500
    elif dataset == 'mushrooms':
        interval = int(len(its_ar[0]) / 2)  # mushrooms
    plot1(its_ar, norms_ar, label_ar, plot_path, dataset, filename=filename, save=1, nworkers=n_ar[0], k_size_compk=k_prime_size_ar[0], k_size_ar_bd=k_size_ar_bd, overlap=overlap_ar[0], interval=interval)
