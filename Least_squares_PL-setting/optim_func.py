import numpy as np
from least_squares_functions_fast import least_squares_grad, least_squares_loss


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