import numpy as np


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
    # omega = float(dim / k) - 1

    if ind: idxs = np.random.choice(dim, k)

    output[idxs] = x[idxs] * float(dim / k)
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
    rand_set = set(all_idx) - set(idx_top)
    rand_idx = np.array(list(rand_set))
    if ind:
        inds_rand = np.random.choice(rand_idx, k)

    inds = np.concatenate((inds_top, inds_rand), axis=0)

    output[inds] = x[inds]  # SRand
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