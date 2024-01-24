def stopping_criterion(sq_norm, eps, it, Nsteps):
    # return (R_k > eps * R_0) and (it <= Nsteps)
    return (it <= Nsteps) and (sq_norm >= eps)