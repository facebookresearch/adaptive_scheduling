# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import time
import scipy
import math

def find_schedule(G, D=1):
    n = len(G)

    Dsq = D**2
    Gsq = G**2

    np.random.seed(42)

    mask = np.zeros(n)
    mask[0] = 1
    mask = torch.tensor(mask)

    x0 = np.array([D/(math.sqrt(n)) for _ in range(n)])

    def func(x_raw):
        if torch.is_tensor(x_raw):
            x = x_raw
        else:
            x = torch.tensor(x_raw, 
                            dtype=torch.float64, 
                            requires_grad=True)
        # Convert to cumulative value
        lamb = x
        lamb_sq = lamb*lamb

        lamb_flip = lamb.flip(dims=(0,))
        lamb_sum = torch.sum(lamb)
        lamb_sq_flip = lamb_flip*lamb_flip
        Gsq_flip = Gsq.flip(dims=(0,))
        t1 = 0.5*Dsq/lamb_sum # Distance error term
        t2 = 0.5/lamb_sum # Gradient error term
        t2 *= torch.sum(Gsq*lamb_sq) 

        inner_cumsum = torch.cumsum(Gsq_flip*lamb_sq_flip, dim=0)
        denom_cumsum = torch.cumsum(lamb_flip, dim=0)
        eval = lamb_flip[1:]*inner_cumsum[1:]/(denom_cumsum[1:]*(denom_cumsum[1:]-lamb_flip[1:]))

        t3 = 0.5*torch.sum(eval)

        fval = (t1+t2+t3)
        fval.backward()

        if torch.is_tensor(x_raw):
            return fval.item()
        else:
            g = list(np.copy(x.grad.numpy()))
            return (fval.item(), g)


    # Test
    fx0, fgx0 = func(x0)

    start = time.time()
    bounds = [(1e-8, np.inf) for _ in range(n)]
    print(f"Starting solve...")
    xopt_inc, fopt, dopt = scipy.optimize.fmin_l_bfgs_b(
        func, x0,
        bounds = bounds,
        iprint = 0,
        factr = 100.0, # High accuracy
        maxls = 100_000,
        maxfun = 100_000,
        pgtol=1e-10,
        m=20,
    )

    end = time.time()
    xopt = xopt_inc
    if dopt['warnflag'] != 0:
        print("WARNING: Nonconvergece")

    return xopt


def find_closed_form_schedule(G, D=1, weights=None):
    T = len(G)
    sched = np.zeros(T)
    G = np.array(G)
    if weights is None:
        weights = G**-2.0
    tail_sums = np.flip(np.cumsum(np.flip(weights)))
    for t in range(T):
        if t == T-1:
            sched[t] = 0
        else:
            sched[t] = weights[t]*tail_sums[t+1]/tail_sums[-1]
    return sched