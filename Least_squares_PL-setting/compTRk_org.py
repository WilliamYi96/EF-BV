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
from IPython import display
from least_squares_functions_fast import *


def stopping_criterion(sq_norm, eps, it, Nsteps):
    # return (R_k > eps * R_0) and (it <= Nsteps)
    return (it <= Nsteps) and (sq_norm >= eps)


####################################################################################
# Top-K
####################################################################################
def top_k_matrix(X, k):
    output = np.zeros(X.shape)
    for i in range(X.shape[0]):
        output[i] = top_k_compressor(X[i], k)
    return output


def top_k_compressor(x, k):
    output = np.zeros(x.shape)
    x_abs = np.abs(x)
    idx = np.argpartition(x_abs, -k)[-k:]  # Indices not sorted
    inds = idx[np.argsort(x_abs[idx])][::-1]
    output[inds] = x[inds]
    return output


####################################################################################
# Rand-K
####################################################################################
def rand_k_matrix(X, k, ind=False):
    # print(X.shape)  # (20, 112)
    output = np.zeros(X.shape)

    dim = X.shape[1]
    idxs = None if ind else np.random.choice(dim, k)

    for i in range(X.shape[0]):
        output[i] = rand_k_compressor(X[i], k, ind, idxs)

    return output


def rand_k_compressor(x, k, ind, idxs=None):
    # RandK compressor with scaling
    output = np.zeros(x.shape)
    dim = x.shape[0]
    omega = float(dim / k) - 1
    scaler = 1.

    if ind: idxs = np.random.choice(dim, k)

    output[idxs] = x[idxs] * float(dim / k) * scaler
    return output


####################################################################################
# scaled Rand-K
####################################################################################
def srand_k_matrix(X, k, ind=False):
    # print(X.shape)  # (20, 112)
    output = np.zeros(X.shape)

    dim = X.shape[1]
    idxs = None if ind else np.random.choice(dim, k)

    for i in range(X.shape[0]):
        output[i] = rand_k_compressor(X[i], k, ind, idxs)

    return output


def srand_k_compressor(x, k, ind, idxs=None):
    # RandK compressor with scaling
    output = np.zeros(x.shape)
    dim = x.shape[0]
    omega = float(dim / k) - 1
    scaler = 1. / (omega + 1)

    if ind: idxs = np.random.choice(dim, k)

    output[idxs] = x[idxs] * float(dim / k) * scaler
    return output


####################################################################################
# mixture of top-k and rand-k
####################################################################################
def mix_k_compressor(x, k, ind=True, inds_rand=None):
    # First: TopK; Next: SRandK
    output = np.zeros(x.shape)
    dim = x.shape[0]
    x_abs = np.abs(x)
    idx_top = np.argpartition(x_abs, -k)[-k:]  # Indices not sorted
    inds_top = idx_top[np.argsort(x_abs[idx_top])][::-1]
    # print(idx_top, inds_top)

    all_idx = np.array([i for i in range(dim)])
    rand_set = set(all_idx) - idx_top
    rand_idx = np.array(list(rand_set))
    if ind:
        inds_rand = np.random.choice(rand_idx, k)

    output[inds_rand] = x[inds_rand]  # SRand
    return output, inds_rand


def mix_k_matrix(X, k, ind=True):
    output = np.zeros(X.shape)

    inds_rand = None
    for i in range(X.shape[0]):
        output[i], inds_rand = mix_k_compressor(X[i], k, ind, inds_rand)
    return output


####################################################################################
# mixture of top-k and rand-k
####################################################################################
def comp_k_compressor(x, kl, ks, ind=True, inds_rand=None):
    # kl: larger K for Top-K selection
    # ks: smaller K for Rand-K selection
    output = np.zeros(x.shape)
    x_abs = np.abs(x)
    idx_top = np.argpartition(x_abs, -kl)[-kl:]  # Indices not sorted
    inds_top = idx_top[np.argsort(x_abs[idx_top])][::-1]
    # print(inds_top)
    # choose ind by default
    if ind:
        inds_rand = np.random.choice(inds_top, ks)
    output[inds_rand] = x[inds_rand]
    return output, inds_rand


def comp_k_matrix(X, kl, ks, ind=True):
    output = np.zeros(X.shape)

    inds_rand = None
    for i in range(X.shape[0]):
        output[i], inds_rand = comp_k_compressor(X[i], kl, ks, ind, inds_rand)
    return output


####################################################################################
####################################################################################
def compute_full_grads(A, x, b, la, n_workers):
    grad_ar = np.zeros((n_workers, x.shape[0]))
    for i in range(n_workers):
        grad_ar[i] = least_squares_grad(x, A[i], b[i], la).copy()
    return grad_ar


def compute_full_funcs(A, x, b, la, n_workers):
    funcs_ar = np.zeros((n_workers, 1))
    for i in range(n_workers):
        funcs_ar[i] = least_squares_loss(x, A[i], b[i], la).copy()
    return funcs_ar


####################################################################################
####################################################################################
def ef21_estimator_ongoing(A, x, b, la, k, h_ar, n_workers, compressor='TopK'):
    grads = compute_full_grads(A, x, b, la, n_workers)
    g_ar_new = np.zeros((n_workers, x.shape[0]))
    delta = grads - h_ar # delta = nabla - h (20, 112)

    compressor_matrix = 0
    if compressor == 'TopK':
        compressor_matrix = top_k_matrix(delta, k)  # dk
    elif compressor == 'SRandK': # scaled Rand-K
        compressor_matrix = srand_k_matrix(delta, k, ind)  # dk

    dk_ar = compressor_matrix
    h_ar = h_ar + dk_ar  # (20, 112)

    # g_ar_new = g_ar + compressor_matrix  # g(k+1) = g(k) + m(k)
    size_value_sent = sys.getsizeof(delta[0, 0])
    # return g_ar_new, size_value_sent, np.mean(grads, axis=0)
    return dk_ar, h_ar, size_value_sent, np.mean(grads, axis=0)


def ef21_ongoing(x_0, A, b, A_0, b_0, stepsize, eps, la, k, n_workers, theta, compressor, Nsteps=100000):
    # Note there is no concept of worker and server in EF21 implementation, as data is sorted at different worker
    print(f"EF21_gd_nw-{n_workers}_k-{k}")
    g_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g = np.mean(g_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    it_ar = [0]
    x = x_0.copy()
    it = 0
    f_ar = [np.mean(compute_full_funcs(A, x_0, b, la, n_workers))]
    p_ar = [0]
    PRINT_EVERY = 1000
    h_ar = np.zeros((n_workers, x.shape[0]))

    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x - stepsize * g
        f_ar.append(np.mean(compute_full_funcs(A, x, b, la, n_workers)))
        dk_ar, h_ar, size_value_sent, grad = ef21_estimator(A, x, b, la, k, h_ar, n_workers, compressor)

        dk = np.mean(dk_ar, axis=0)
        h = np.mean(h_ar, axis=0)
        g = g + dk
        print(dk[:2], h[:2], g[:2])

        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        p_ar.append((stepsize / theta) * np.linalg.norm(x=g - grad, ord=2) ** 2)
        it += 1
        it_ar.append(it * k * size_value_sent)
        if it % PRINT_EVERY == 0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1])
    return np.array(it_ar), np.array(sq_norm_ar), x, np.array(f_ar), np.array(p_ar)


####################################################################################
####################################################################################
def ef21_estimator(A, x, b, la, k, g_ar, h_ar, n_workers, compressor='TopK'):
    # grads = compute_full_grads(A, x, b, la, n_workers)
    # g_ar_new = np.zeros((n_workers, x.shape[0]))
    # delta = grads - g_ar
    #
    # compressor_matrix = 0
    # if compressor == 'TopK':
    #     compressor_matrix = top_k_matrix(delta, k)  # m(k)
    # elif compressor == 'SRandK': # scaled Rand-K
    #     compressor_matrix = srand_k_matrix(delta, k, ind)  # m(k)
    #
    # g_ar_new = g_ar + compressor_matrix  # g(k+1) = g(k) + m(k)
    # size_value_sent = sys.getsizeof(delta[0, 0])
    # return g_ar_new, size_value_sent, np.mean(grads, axis=0)

    # new implementation
    grads = compute_full_grads(A, x, b, la, n_workers)
    g_ar_new = np.zeros((n_workers, x.shape[0]))
    delta = grads - h_ar

    d_ar = 0
    if compressor == 'TopK':
        d_ar = top_k_matrix(delta, k)
    elif compressor == 'SRandK': # scaled Rand-K
        d_ar = srand_k_matrix(delta, k, ind)
    elif compressor == 'CompK':
        d_ar = comp_k_matrix(delta, 2*k, k, ind)

    g_ar_new = g_ar + d_ar
    h_ar_new = h_ar + d_ar
    size_value_sent = sys.getsizeof(delta[0, 0])
    return g_ar_new, h_ar_new, size_value_sent, np.mean(grads, axis=0)


def ef21(x_0, A, b, A_0, b_0, stepsize, eps, la, k, n_workers, theta, compressor, Nsteps=100000):
    # Note there is no concept of worker and server in EF21 implementation, as data is sorted at different worker
    print(f"EF21_gd_nw-{n_workers}_k-{k}")
    g_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g = np.mean(g_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    it_ar = [0]
    x = x_0.copy()
    it = 0
    f_ar = [np.mean(compute_full_funcs(A, x_0, b, la, n_workers))]
    p_ar = [0]
    PRINT_EVERY = 1000
    # h_ar = np.zeros((n_workers, x.shape[0]))
    h_ar = g_ar.copy()

    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x - stepsize * g
        f_ar.append(np.mean(compute_full_funcs(A, x, b, la, n_workers)))
        g_ar, h_ar, size_value_sent, grad = ef21_estimator(A, x, b, la, k, g_ar, h_ar, n_workers, compressor=compressor)
        g = np.mean(g_ar, axis=0)  # g^(k+1) = mean(g_i(k) + m_i(k))
        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        p_ar.append((stepsize / theta) * np.linalg.norm(x=g - grad, ord=2) ** 2)
        it += 1
        it_ar.append(it * k * size_value_sent)
        if it % PRINT_EVERY == 0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1])
    return np.array(it_ar), np.array(sq_norm_ar), x, np.array(f_ar), np.array(p_ar)


####################################################################################
####################################################################################
def ef22_estimator(A, x, b, la, k, g_ar, h_ar, n_workers, compressor='CompK', lamd_ar=0, nu_ar=0):
    grads = compute_full_grads(A, x, b, la, n_workers)  # (20, 112)
    g_ar_new = np.zeros((n_workers, x.shape[0]))  # (20, 112)
    delta = grads - h_ar  # delta = nabla - h (20, 112)

    d_ar = 0  # d_k
    if compressor == 'MixK': # mixture of Top-K and Rand-K
        d_ar = mix_k_matrix(delta, k, ind)
    elif compressor == 'CompK':
        kl, ks = 2 * k, k
        d_ar = comp_k_matrix(delta, kl, ks, ind)
        # d_ar = top_k_matrix(delta, k)

    # print(nu_ar, lamd_ar)  # [0.17908771] [0.00896807]
    g_ar_new = g_ar + nu_ar * d_ar
    h_ar_new = h_ar + lamd_ar * d_ar
    # g_ar_new = g_ar + d_ar
    # h_ar_new = h_ar + d_ar
    size_value_sent = sys.getsizeof(delta[0, 0])
    return g_ar_new, h_ar_new, size_value_sent, np.mean(grads, axis=0)


def ef22(x_0, A, b, A_0, b_0, stepsize, eps, la, k, n_workers, theta, compressor, Nsteps=100000, lamd_ar=0, nu_ar=0):
    # Note there is no concept of worker and server in EF21 implementation, as data is sorted at different worker
    print(f"EF22_gd_nw-{n_workers}_k-{k}")
    g_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g = np.mean(g_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    it_ar = [0]
    x = x_0.copy()
    it = 0
    f_ar = [np.mean(compute_full_funcs(A, x_0, b, la, n_workers))]
    p_ar = [0]
    PRINT_EVERY = 1000
    # h_ar = np.zeros((n_workers, x.shape[0]))  # initialize h_0 = 1 / n sum(h_i^0)
    h_ar = g_ar.copy()

    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x - stepsize * g
        f_ar.append(np.mean(compute_full_funcs(A, x, b, la, n_workers)))
        g_ar, h_ar, size_value_sent, grad = ef22_estimator(A, x, b, la, k, g_ar, h_ar, n_workers,
                                                           compressor=compressor, lamd_ar=lamd_ar, nu_ar=nu_ar)
        g = np.mean(g_ar, axis=0)  # g^(k+1) = mean(g_i(k) + m_i(k))
        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        p_ar.append((stepsize / theta) * np.linalg.norm(x=g - grad, ord=2) ** 2)
        it += 1
        it_ar.append(it * k * size_value_sent)
        if it % PRINT_EVERY == 0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1])
    return np.array(it_ar), np.array(sq_norm_ar), x, np.array(f_ar), np.array(p_ar)


####################################################################################
####################################################################################
def diana_estimator(A, x, b, la, k, g_ar, h_ar, n_workers, compressor='RandK', lamd_ar=0):
    grads = compute_full_grads(A, x, b, la, n_workers)
    g_ar_new = np.zeros((n_workers, x.shape[0]))
    delta = grads - h_ar

    compressor_matrix = 0
    if compressor == 'TopK':
        compressor_matrix = top_k_matrix(delta, k)
    elif compressor == 'MixK':
        compressor_matrix = mix_k_matrix(delta, k)
    elif compressor == 'SRandK': # scaled Rand-K
        compressor_matrix = srand_k_matrix(delta, k, ind)
    elif compressor == 'MixK': # mixture of Top-K and Rand-K
        compressor_matrix = mix_k_matrix(delta, k, ind)
    elif compressor == 'CompK':
        kl, ks = 2*k, k
        compressor_matrix = comp_k_matrix(delta, kl, ks, ind)
    elif compressor == 'RandK':
        compressor_matrix = rand_k_compressor(delta, k, ind)

    g_ar_new = g_ar + lamd_ar * compressor_matrix
    h_ar_new = h_ar + compressor_matrix
    size_value_sent = sys.getsizeof(delta[0, 0])
    return g_ar_new, h_ar_new, size_value_sent, np.mean(grads, axis=0)


def diana(x_0, A, b, A_0, b_0, stepsize, eps, la, k, n_workers, theta, compressor, Nsteps=100000, lamd_ar=0, nu_ar=0):
    # Note there is no concept of worker and server in EF21 implementation, as data is sorted at different worker
    print(f"EF22_gd_nw-{n_workers}_k-{k}")
    g_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g = np.mean(g_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    it_ar = [0]
    x = x_0.copy()
    it = 0
    f_ar = [np.mean(compute_full_funcs(A, x_0, b, la, n_workers))]
    p_ar = [0]
    PRINT_EVERY = 1000
    h = 0  # initialize h_0 = 1 / n sum(h_i^0)

    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x - stepsize * g
        f_ar.append(np.mean(compute_full_funcs(A, x, b, la, n_workers)))
        g_ar, h_ar, size_value_sent, grad = diana_estimator(A, x, b, la, k, g_ar, n_workers, compressor, lamd_ar=lamd_ar)
        g = np.mean(g_ar, axis=0)  # g^(k+1) = mean(g_i(k) + m_i(k))
        # m = np.mean(m_ar, axis=0)  # aggregate received messages
        # g = h + m  # the only difference between DIANA and EF22
        # h = h + lamd_ar * m
        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        p_ar.append((stepsize / theta) * np.linalg.norm(x=g - grad, ord=2) ** 2)
        it += 1
        it_ar.append(it * k * size_value_sent)
        if it % PRINT_EVERY == 0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1])
    return np.array(it_ar), np.array(sq_norm_ar), x, np.array(f_ar), np.array(p_ar)


def save_data(its, f_grad_norms, x_solution, f_ar, p_ar, k_size, experiment_name, project_path, dataset):
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
dataset = args.dataset
loss_func = "least-squares"
data_path = project_path + "data_{0}/".format(dataset)
if not os.path.exists(data_path):
    os.mkdir(data_path)

data_info = np.load(data_path + 'data_info.npy')

la = 0  # no regularizer by default

X_0 = np.load(data_path + 'X.npy')  # whole dateset
y_0 = np.load(data_path + 'y.npy')
n_0, d_0 = X_0.shape  # mushrooms: (8124, 112)

hess_f_0 = (1 / (n_0)) * (X_0.T @ X_0) + 2 * la * np.eye(d_0)

L_0 = np.max(np.linalg.eigvals(hess_f_0))
# L_0 = L_0.astype(np.float)
mu_0 = np.linalg.svd(X_0)[1][-1] ** 2
print(f"L_0 = {L_0}, mu_0 = {mu_0}")  # (10.34+0j), 1.65e-28

ind = True

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

    ####################################################################################
    # Choose compressor and method
    ####################################################################################
    if compressor == 'TopK' or compressor == 'SRandK':
        al_ar = k_ar / d_0
        # For EF21 fair comparison
        eta = np.sqrt(1 - al_ar)
        omega = 0
    elif compressor == 'RandK':
        omega = d_0 / k_ar - 1
    elif compressor == 'MixK':
        assert k_ar <= d_0 / 2
        eta = (d_0 - 2 * k_ar) / np.sqrt((d_0-k_ar) * d_0)
        omega = k_ar * (d_0 - 2*k_ar) / ((d_0 - k_ar) * d_0)
        al_ar = 2 * k_ar / d_0
        omega_av = omega / num_workers if ind else omega
        print(eta, omega, al_ar)
    elif compressor == 'CompK':
        # kl, ks = 2 * k_ar, k_ar
        # kl, ks = int(np.sqrt(d_0)) * k_ar, k_ar
        kl, ks = 5 * k_ar, k_ar  # TODO
        eta = np.sqrt((d_0 - kl) / d_0)
        omega = (kl - ks) / ks
        omega_av = omega / num_workers if ind else omega
        al_ar = 2 * k_ar / d_0
        print(eta, omega, omega_av)  # [0.99103121] [1.] [1.]

    Lt = np.sqrt(np.mean(L ** 2))

    if args.method == 'EF22':
        # only support MixK and CompK for now
        lamd_ar = np.minimum(1, (1 - eta) / ((1 - eta) ** 2 + omega))
        nu_ar = np.minimum(1, (1 - eta) / ((1 - eta) ** 2 + omega_av))
        r = (1 - lamd_ar + lamd_ar * eta)**2 + lamd_ar**2 * omega
        r_av = (1 - nu_ar + nu_ar * eta)**2 + nu_ar**2 * omega_av
        ss = np.sqrt((1 + r) / (2 * r)) - 1
        theta_ar = ss * (1 + ss) * r / r_av
        gamma = 1. / (L_0 + Lt * np.sqrt(r_av / r) * (1. / ss)) * factor
        # print(f'EF22 key params: lamd: {lamd_ar}, nu: {nu_ar}, r: {r_av}, r_av: {r_av}, '
        #       f'gamma: {gamma}, s: {ss}, theta: {theta_ar}')

        # This is for EF21 comparision
        t_ar = -1 + np.sqrt(1 / (1 - al_ar))
        theta_ar1 = 1 - (1 - al_ar) * (1 + t_ar)
        beta_ar1 = (1 - al_ar) * (1 + 1 / t_ar)
        gamma1 = np.minimum((1 / (L_0 + Lt * np.sqrt(2 * beta_ar1 / theta_ar1))), theta_ar1 / (2 * mu_0)) * factor
        if gamma > gamma1:
            print(f'EF22 improves EF21 by {gamma - gamma1}')
        else:
            print(f'EF21 has better learning rate than EF22 by {gamma1-gamma}')
        # assert 0
        print(lamd_ar, nu_ar, r, r_av, ss, theta_ar, gamma)
        assert 0

    elif args.method == 'EF21':
        # only support TopK and SRandK for now
        # if k_ar[0] == d_0:
        #     theta_ar = 1 + 0 * k_ar
        #     beta_ar = 0 * k_ar
        # else:
        #     t_ar = -1 + np.sqrt(1 / (1 - al_ar))
        #     theta_ar = 1 - (1 - al_ar) * (1 + t_ar)
        #     beta_ar = (1 - al_ar) * (1 + 1 / t_ar)
        # gamma = np.minimum((1 / (L_0 + Lt * np.sqrt(2 * beta_ar / theta_ar))), theta_ar / (2 * mu_0)) * factor
        # if theta_ar / (2 * mu_0) < (1 / (L_0 + Lt * np.sqrt(2 * beta_ar / theta_ar))):
        #     print(f"PL stepsize works! Improvement in "
        #           f"{(1 / (L_0 + Lt * np.sqrt(2 * beta_ar / theta_ar))) / (theta_ar / (2 * mu_0))} times!")
        # print(theta_ar, beta_ar, gamma)

        # Consider the following new implementation
        lamd_ar = np.minimum(1, (1 - eta) / ((1 - eta) ** 2 + omega))
        nu_ar = np.minimum(1, (1 - eta) / ((1 - eta) ** 2 + omega))
        r = (1 - lamd_ar + lamd_ar * eta)**2 + lamd_ar**2 * omega
        r_av = (1 - nu_ar + nu_ar * eta)**2 + nu_ar**2 * omega
        ss = np.sqrt((1 + r) / (2 * r)) - 1
        theta_ar = ss * (1 + ss) * r / r_av
        gamma = 1. / (L_0 + Lt * np.sqrt(r_av / r) * (1. / ss)) * factor
        print(lamd_ar, nu_ar, r, r_av, ss, theta_ar, gamma)

    elif args.method == 'DIANA':
        gamma = 1 / (1 + 6 * omega / num_workers) / L_0 * factor
        theta_ar = [0]
        lamd_ar = 1 / (omega + 1)
        print(f'lamda: {lamd_ar}; gamma: {gamma}')
        # assert 0

    print(f'step_size_{args.method}_tpc: ', gamma, compressor)

    experiment_name = "{3}-full-grad_nw-{0}_{1}x_{2}".format(n_ar[i], factor, compressor, args.method)

    for k in range(len(k_ar)):
        if args.method == 'DIANA':
            results = diana(x_0, X, y, X_0, y_0, gamma[k], eps, la, k_ar[k], n_ar[i],
                           theta_ar[k], compressor=compressor, Nsteps=nsteps, lamd_ar=lamd_ar)
        elif args.method == 'EF21':
            results = ef21(x_0, X, y, X_0, y_0, gamma[k], eps, la, k_ar[k], n_ar[i],
                           theta_ar[k], compressor=compressor, Nsteps=nsteps)
        elif args.method == 'EF22':
            results = ef22(x_0, X, y, X_0, y_0, gamma[k], eps, la, k_ar[k], n_ar[i],
                           theta_ar[k], compressor=compressor, Nsteps=nsteps, lamd_ar=lamd_ar, nu_ar=nu_ar)
        print(experiment_name + f" with k={k_ar[k]} finished in {results[0].shape[0]} iterations")
        its = results[0]
        norms = results[1]
        sols = results[2]
        f_ar = results[3]
        p_ar = results[4]

        save_data(its, norms, sols, f_ar, p_ar, k_ar[k], experiment_name, project_path, dataset)
