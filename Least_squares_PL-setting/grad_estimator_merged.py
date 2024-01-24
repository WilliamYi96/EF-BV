from optim_func import compute_full_funcs, compute_full_grads
import numpy as np
from utils import stopping_criterion
from compressors import top_k_matrix, rand_k_matrix, srand_k_matrix, mix_k_matrix, comp_k_matrix
import sys
from IPython import display


####################################################################################
####################################################################################
def grad_estimator(A, x, b, la, k, g_ar, h_ar, n_workers, compressor='TopK', ratio=2, lamd=1, nu=1, ind=True):
    grads = compute_full_grads(A, x, b, la, n_workers)
    g_ar_new = np.zeros((n_workers, x.shape[0]))
    delta = grads - h_ar

    d_ar = 0
    if compressor == 'RandK':
        d_ar = rand_k_matrix(delta, k, ind)
    elif compressor == 'TopK':
        d_ar = top_k_matrix(delta, k)
    elif compressor == 'SRandK':  # scaled Rand-K
        d_ar = srand_k_matrix(delta, k, ind)
    elif compressor == 'CompK':
        d_ar = comp_k_matrix(delta, ratio * k, k, ind)
    elif compressor == 'MixK':
        d_ar = mix_k_matrix(delta, k, ind)
    elif compressor == 'CompK':
        kl, ks = ratio * k, k
        d_ar = comp_k_matrix(delta, kl, ks, ind)

    g_ar_new = h_ar + nu * d_ar
    h_ar_new = h_ar + lamd * d_ar
    size_value_sent = 2 * sys.getsizeof(delta[0, 0]) if compressor == 'MixK' else sys.getsizeof(delta[0, 0])

    return g_ar_new, h_ar_new, size_value_sent, np.mean(grads, axis=0)


def methods(x_0, A, b, A_0, b_0, stepsize, eps, la, k, n_workers,
            compressor, Nsteps=100000, ratio=1, lamd=1, nu=1, method='DIANA', ind=True):
    # This methods will not compute the relative values of f(x)-f(x*)
    print(f"{method}_gd_nw-{n_workers}_k-{k}")
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
    theta = 1

    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x - stepsize * g
        # print(f_ar[-1])
        f_ar.append(np.mean(compute_full_funcs(A, x, b, la, n_workers)))
        g_ar, h_ar, size_value_sent, grad = grad_estimator(A, x, b, la, k, g_ar, h_ar, n_workers, compressor=compressor,
                                                           ratio=ratio, lamd=lamd, nu=nu, ind=ind)
        g = np.mean(g_ar, axis=0)  # g^(k+1) = mean(g_i(k) + m_i(k))
        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        p_ar.append((stepsize / theta) * np.linalg.norm(x=g - grad, ord=2) ** 2)
        it += 1
        it_ar.append(it * k * size_value_sent)
        if it % PRINT_EVERY == 0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1], f_ar[-1])
    return np.array(it_ar), np.array(sq_norm_ar), x, np.array(f_ar), np.array(p_ar)


def gd(x_0, A, b, A_0, b_0, stepsize, eps, la, n_workers, Nsteps=100000):
    # Compute optimal x*
    print(f'Running gradient descent to obtain optimal value...')
    x = x_0.copy()
    it = 0
    # Compute initial sq_norm_ar
    g_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g = np.mean(g_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    PRINT_EVERY = 1000

    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        g_ar = compute_full_grads(A, x, b, la, n_workers)
        g = np.mean(g_ar, axis=0)
        x = x - stepsize * g
        sq_norm_ar.append(np.linalg.norm(x=g, ord=2) ** 2)
        it += 1
        if it % PRINT_EVERY == 0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1])
    print(f'Finish GD running')
    fx = np.mean(compute_full_funcs(A, x, b, la, n_workers), axis=0)
    return x, fx