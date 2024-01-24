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
import datetime

from grad_estimator_merged import methods, gd


def save_data(its, f_grad_norms, x_solution, f_ar, p_ar, k_size, experiment_name, project_path, dataset, diff_ar):
    experiment = '{0}_{1}'.format(experiment_name, k_size)
    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)

    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    np.save(logs_path + 'iteration' + '_' + experiment, np.array(its))
    np.save(logs_path + 'solution' + '_' + experiment, x_solution)
    np.save(logs_path + 'norms' + '_' + experiment, np.array(f_grad_norms))
    np.save(logs_path + 'f_ar' + '_' + experiment, np.array(f_ar))
    np.save(logs_path + 'p_ar' + '_' + experiment, np.array(p_ar))
    np.save(logs_path + 'diff_ar' + '_' + experiment, np.array(diff_ar))


user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"

parser = argparse.ArgumentParser(description='Run top-k algorithm')  # refactor to support different algorithms
parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=None,
                    help='Maximum number of iteration')
parser.add_argument('--k', action='store', dest='k', type=int, default=1, help='Sparcification parameter')
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=20,
                    help='Number of workers that will be used')
parser.add_argument('--factor', action='store', dest='factor', type=int, default=1, help='Stepsize factor')
parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-5, help='tolerance')  # threshold
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms', help='name_of_dataset')
parser.add_argument('--ind', action='store_true', help='independent randomization')
parser.add_argument('--compressor', type=str, default='SRandK', help='Select from TopK, MixK, SRandK, CompK')
parser.add_argument('--method', type=str, default='EF22', help='Select from EF22, EF21, DIANA')
parser.add_argument('--mdouble', action='store_true', help='Top2K for EF21, Rand2K for DIANA')
parser.add_argument('--ratio', type=int, default=2, help='k value instead of the k prime value')
parser.add_argument('--overlap', type=int, default=1, help='overlap rate')

args = parser.parse_args()
nsteps = args.max_it
k_size = 2 * args.k if args.mdouble else args.k
num_workers = args.num_workers

ind = args.ind
compressor = args.compressor
n_ar = np.array([num_workers])
k_ar = np.array([k_size])
factor = args.factor
eps = args.tol
ratio = args.ratio
dataset = args.dataset
loss_func = "least-squares"
data_path = project_path + "data_{0}_{1}/".format(dataset, args.overlap)
if not os.path.exists(data_path):
    os.mkdir(data_path)

data_info = np.load(data_path + 'data_info.npy')

la = 0  # no regularizer by default

X_0 = np.load(data_path + 'X.npy')  # whole dateset
y_0 = np.load(data_path + 'y.npy')
n_0, d_0 = X_0.shape  # mushrooms: (8124, 112)

hess_f_0 = (1 / (n_0)) * (X_0.T @ X_0) + 2 * la * np.eye(d_0)

L_0 = np.max(np.linalg.eigvals(hess_f_0))
mu_0 = np.linalg.svd(X_0)[1][-1] ** 2

# L_0 = L_0.astype(np.float)
print(f"L_0 = {L_0}, mu_0 = {mu_0}")  # (10.34+0j), 1.65e-28

# distribute dataset to n_workers
for i in range(len(n_ar)):
    # c = subprocess.call(f"python3 generate_data.py --dataset mushrooms --num_starts 1 --num_workers {n_ar[i]}
    # --loss_func log-reg --is_homogeneous 0", shell=True)
    X = []
    y = []
    L = np.zeros(n_ar[i])
    n = np.zeros(n_ar[i], dtype=int)
    d = np.zeros(n_ar[i], dtype=int)

    for j in range(n_ar[i]):
        X.append(np.load(data_path + 'X_{0}_nw{1}_{2}.npy'.format(dataset, n_ar[i], j)))
        y.append(np.load(data_path + 'y_{0}_nw{1}_{2}.npy'.format(dataset, n_ar[i], j)))
        n[j], d[j] = X[j].shape

        currentDT = datetime.datetime.now()
        # print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
        # print (X[j].shape)

        hess_f_j = (1 / (n[j])) * (X[j].T @ X[j]) + 2 * la * np.eye(d[j])
        L[j] = np.max(np.linalg.eigvals(hess_f_j))

    L = L.astype(np.float)

    if not os.path.isfile(data_path + 'w_init_{0}.npy'.format(loss_func)):
        # create a new w_0
        x_0 = np.random.normal(loc=0.0, scale=2.0, size=d_0)
        np.save(data_path + 'w_init_{0}.npy'.format(loss_func), x_0)
        x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))
    else:
        # load existing w_0
        x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))

    ##################
    # x_0 = np.ones(d_0)
    ##################

    num_workers = [20, 50, 100, 1000, 10000]
    ratios = [1, 2, 5, 10, 56, 112]

    for num_worker in num_workers:
        for ratio in ratios:
            kl, ks = ratio * k_ar, k_ar
            eta = np.sqrt((d_0 - kl) / d_0)
            omega = (kl - ks) / ks
            omega_av = omega / num_worker if ind else omega
            al_ar = 1 * k_ar / d_0
            # print(eta, omega, omega_av)  # [0.99103121] [1.] [1.]

            Lt = np.sqrt(np.mean(L ** 2))

            if args.method == 'EF22':
                lamd_ar = np.minimum(1, (1 - eta) / ((1 - eta) ** 2 + omega))
                nu_ar = np.minimum(1, (1 - eta) / ((1 - eta) ** 2 + omega_av))
                r = (1 - lamd_ar + lamd_ar * eta)**2 + lamd_ar**2 * omega
                r_av = (1 - nu_ar + nu_ar * eta)**2 + nu_ar**2 * omega_av
                ss = np.sqrt((1 + r) / (2 * r)) - 1
                theta_ar = ss * (1 + ss) * r / r_av
                gamma = 1. / (L_0 + Lt * np.sqrt(r_av / r) * (1. / ss)) * factor
            elif args.method == 'EF21':
                lamd_ar = np.minimum(1, (1 - eta) / ((1 - eta) ** 2 + omega))
                nu_ar = np.minimum(1, (1 - eta) / ((1 - eta) ** 2 + omega))
                r = (1 - lamd_ar + lamd_ar * eta)**2 + lamd_ar**2 * omega
                r_av = (1 - nu_ar + nu_ar * eta)**2 + nu_ar**2 * omega
                ss = np.sqrt((1 + r) / (2 * r)) - 1
                theta_ar = ss * (1 + ss) * r / r_av
                gamma = 1. / (L_0 + Lt * np.sqrt(r_av / r) * (1. / ss)) * factor
                # lamd = nu = 1

            print(num_worker, ratio, eta, omega, omega_av, lamd_ar, nu_ar, r, r_av, ss, gamma)

