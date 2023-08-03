# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, minimize, SR1
import pdb
import math
import numpy.random
import time

import torch

n = 1000

G = 1.0
D = 1.0

Gsq = G**2
Dsq = D**2

numpy.random.seed(42)

mask = np.zeros(n)
mask[0] = 1
mask = torch.tensor(mask)

def lamb_from_increments_torch(x):
    xmod = x.sub(x*mask) # Set first entry to 0
    v = torch.exp(-xmod)
    cexp = torch.cumprod(v, dim=0)
    cexp_shift = cexp * x[0]
    #pdb.set_trace()
    return cexp_shift

def lamb_from_increments(xraw):
    if not torch.is_tensor(xraw):
        x = torch.tensor(xraw, dtype=torch.float64)
    else:
        x = xraw
    result = lamb_from_increments_torch(x)

    if torch.is_tensor(xraw):
        return result
    else:
        return result.numpy()


def lamb_to_increments(yraw):
    if not torch.is_tensor(yraw):
        y = torch.tensor(yraw, dtype=torch.float64)
    else:
        y = yraw
    def inv_cum_prod(v):
        return torch.exp(torch.diff(torch.log(v)))
    log_incs = -torch.log(inv_cum_prod(y))
    result = torch.concatenate(
        (torch.tensor([y[0]]), log_incs))
    if torch.is_tensor(yraw):
        return result
    else:
        return result.numpy()

y0 = np.flip(np.cumsum(np.abs(numpy.random.normal(size=n))))/n
x0 = lamb_to_increments(y0)

assert np.all(np.isclose(lamb_from_increments(x0), y0))

def func(x_raw):
    if torch.is_tensor(x_raw):
        x = x_raw
    else:
        x = torch.tensor(x_raw, 
                         dtype=torch.float64, 
                         requires_grad=True)

    lamb = lamb_from_increments_torch(x)

    lamb_flip = lamb.flip(dims=(0,))
    lamb_sum = torch.sum(lamb)
    lamb_sq_flip = lamb_flip*lamb_flip
    t1 = 0.5*Dsq/lamb_sum # Distance error term
    t2 = 0.5*Gsq/lamb_sum # Gradient error term
    t2 *= torch.sum(lamb_sq_flip) 

    inner_cumsum = torch.cumsum(lamb_sq_flip, dim=0)
    denom_cumsum = torch.cumsum(lamb_flip, dim=0)
    eval = lamb_flip[1:]*inner_cumsum[1:]/(denom_cumsum[1:]*(denom_cumsum[1:]-lamb_flip[1:]))

    t3 = 0.5*Gsq*torch.sum(eval)

    fval = (t1+t2+t3) #/max(G/D,D/G)

    fval.backward()

    if torch.is_tensor(x_raw):
        return fval.item()
    else:
        g = list(np.copy(x.grad.numpy()))
        return (fval.item(), g)


# Test
fx0, fgx0 = func(x0)

start = time.time()
bounds = [(1e-12, np.inf)] + [(0, 10) for _ in range(n-1)]
print(f"Starting solve...")
xopt_inc, fopt, dopt = scipy.optimize.fmin_l_bfgs_b(
    func, x0,
    bounds = bounds,
    iprint = 0,
    factr = 10.0, # High accuracy
    maxls = 100000,
    maxfun = 100000,
    pgtol=1e-10,
    m=20,
)

end = time.time()
xopt = lamb_from_increments(xopt_inc)
assert dopt['warnflag'] == 0

print(f"Time taken: {end - start}")
print(f"Steps to convergence: {dopt['funcalls']}")
#print(f"grad: {dopt['grad']}")
#print(xopt)
print(f"xopt[0]: {xopt[0]}")
print(f"xopt[-1]: {xopt[-1]}")
print(f"xopt[0]/xopt[-1]: {xopt[0]/xopt[-1]}")
print(f"fval: {fopt}")
print(f"fval * sqrt(n): {fopt * math.sqrt(n)} ")

def func1d(x_raw):
    eta = torch.tensor(x_raw, 
                           dtype=torch.float64, 
                           requires_grad=True)
    t1 = Dsq/(2*n*eta)
    t2 = Gsq*eta/2
    t3 = (Gsq*eta/2)*torch.sum(1/torch.arange(1, n))
    fval = (t1+t2+t3)#/max(G/D,D/G)
    fval.backward()

    if torch.is_tensor(x_raw):
        return fval.item()
    else:
        g = list(np.copy(eta.grad.numpy()))
        return (fval.item(), g)

xopt_1d, fopt_1d, dopt_1d = scipy.optimize.fmin_l_bfgs_b(
    func1d, np.array([y0[0]]), bounds = [(1e-8, 100)],
    iprint = 0
)
assert dopt_1d['warnflag'] == 0
xopt_1d = xopt_1d[0]

print(f"1D grad: {dopt_1d['grad']}")
print(f"1D Steps to convergence: {dopt_1d['funcalls']}")
#print(f"grad: {dopt_1d['grad']}")
print(f"eta 1d: {xopt_1d}")
print(f"1D fval: {fopt_1d}")

theory_eta = D/(G*math.sqrt(n*(2+math.log(n-1))))
theory1d = (D*G*math.sqrt(2+math.log(n-1))/math.sqrt(n))#/max(G/D,D/G)
print(f"Theory eta: {theory_eta}")
print(f"theory 1d fval: {theory1d}")

print(f"1d/full ratio: {fopt_1d/fopt}")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.titlesize'] = 5
mpl.rcParams['axes.labelsize'] = 5
mpl.rcParams['font.size'] = 4.2
mpl.rcParams['legend.fontsize'] = 4.2
linewidth = '0.2'
mpl.rcParams['lines.markersize'] = 1.0
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.linewidth'] = linewidth
mpl.rcParams['xtick.major.width'] = linewidth
mpl.rcParams['ytick.major.width'] = linewidth

fig = plt.figure(figsize=(4, 3))

ax = fig.add_subplot(2, 1, 1)
plt.tight_layout()
ax.set_xlabel('k')
ax.set_ylabel('lamb')
ax.set_title(f"Optimal step size sequence v.s. optimal flat Dsq={D} Gsq={G}")
ax.plot(range(1, n+1), xopt, 'k')
ax.hlines(y=xopt_1d, xmin=1, xmax=n, color='r')
ax.hlines(y=D/(G*math.sqrt(n)), xmin=1, xmax=n, color='b')
#ax.set_yscale('log')
plt.tight_layout()

ax = fig.add_subplot(2, 1, 2)
plt.tight_layout()
ax.set_xlabel('k')
ax.set_ylabel('lamb')
ax.set_title(f"Optimal step size sequence v.s. optimal flat D={D} G={G}")
ax.plot(range(1, n+1), xopt, 'k')
ax.hlines(y=xopt_1d, xmin=1, xmax=n, color='r')
ax.hlines(y=D/(G*math.sqrt(n)), xmin=1, xmax=n, color='b')
ax.set_yscale('log')
plt.tight_layout()

fname = "lamb_lbfgs.png"
plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
print(f"Saved {fname}")
plt.close()
plt.close('all')