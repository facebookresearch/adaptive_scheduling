# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, minimize, SR1
import pdb
import math
import numpy.random
import time
from scipy.interpolate import UnivariateSpline, splrep, BSpline, splev

import torch

n = 500

#G = torch.tensor([1-i/(n+1) for i in range(n)])
G = torch.tensor([1.0 for i in range(n)])

# CIFAR10 approx pattern
#G = torch.concatenate((1.0*torch.ones(7*n//8), 0.5*torch.ones(n//8)))

# Imagenet like
#G = torch.tensor([1.0 + 1.0*i/n for i in range(n)])
#G = torch.tensor([1.0 - 0.5*i/n for i in range(n)])

#G = torch.tensor([min(0.1, 1.0/math.sqrt(i+1)) for i in range(n)])

#G = torch.concatenate((10.0*torch.tensor([1-i/(n+1) for i in range(n//4)]), 1.0*torch.tensor([1-i/(n+1) for i in range(n//4)]), 0.1*torch.ones(n//2)))

G = torch.concatenate((
    torch.tensor([max(1, 10*(1-i/(n//10+1))) for i in range(n//10)]), 
    torch.tensor([1.0 for i in range(9*n//10)])))

# This one gives very promising shapes!
# It gives a learning rate warmup at the begining,
# with a fall-off thats more gradual and cosine like.
# G = torch.concatenate((
#     torch.tensor([max(1, 10*(1-i/(n//10+1))) for i in range(n//10)]), 
#     torch.tensor([1.0 + (i/(9*n//10)) for i in range(9*n//10)])))

# No warmup version

#G = torch.tensor([1.0 + 1.0*i/n for i in range(n)])
# G = torch.concatenate((
#     torch.tensor([((i+1)/(n//100+1)) for i in range(n//100)]), 
#     torch.tensor([1.0 + (i/((99*n)//100)) for i in range((99*n)//100)])))

# G = torch.concatenate((
#     torch.tensor([max(1, 2*(1-i/(n//10+1))) for i in range(n//10)]), 
#     torch.tensor([1.0 - 0.3*(i/(9*n//10)) for i in range(9*n//10)])))

# spl = splrep(x=[0, n//10, n], y=[10, 1, 2], k=2)
# spl(range(n))

#G = torch.tensor(scipy.ndimage.gaussian_filter1d(G, sigma=30))

D = 1.0

Dsq = D**2
Gsq = G**2

numpy.random.seed(42)

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
bounds = [(1e-12, np.inf) for _ in range(n)]
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
xopt = xopt_inc
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

cosine_curve = [D/(math.sqrt(n)) * 0.5 * (1 + math.cos((i/n) * math.pi)) for i in range(n)]

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

fig = plt.figure(figsize=(4, 5))

ax = fig.add_subplot(3, 1, 1)
plt.tight_layout()
ax.set_xlabel('k')
ax.set_ylabel('lamb')
ax.set_title(f"Optimal step size sequence (final={xopt[-1]})")
ax.plot(range(1, n+1), xopt, 'k')
ax.plot(range(1, n+1), [(1-i/(n+1))*D/(math.sqrt(n)) for i in range(n)], color='purple')
ax.plot(range(1, n+1), cosine_curve, color='r')
ax.hlines(y=D/(math.sqrt(n)), xmin=1, xmax=n, color='b')
ax.hlines(y=(1-n/(n+1))*D/(math.sqrt(n)), xmin=1, xmax=n, color='y')
ax.plot(range(1, n+1), [((1-i/(n+1))**0.5)*D/(math.sqrt(n)) for i in range(n)], color='pink')
plt.tight_layout()

ax = fig.add_subplot(3, 1, 2)
plt.tight_layout()
ax.set_xlabel('k')
ax.set_ylabel('lamb')
ax.set_title(f"Optimal step size sequence")
ax.plot(range(1, n+1), xopt, 'k')
ax.plot(range(1, n+1), [(1-i/(n+1))*D/(math.sqrt(n)) for i in range(n)], color='purple')
ax.plot(range(1, n+1), cosine_curve, color='r')
ax.plot(range(1, n+1), [((1-i/(n+1))**0.5)*D/(math.sqrt(n)) for i in range(n)], color='pink')
ax.hlines(y=D/(math.sqrt(n)), xmin=1, xmax=n, color='b')
ax.hlines(y=(1-n/(n+1))*D/(math.sqrt(n)), xmin=1, xmax=n, color='y')
ax.set_yscale('log')
plt.tight_layout()

ax = fig.add_subplot(3, 1, 3)
plt.tight_layout()
ax.set_xlabel('k')
ax.set_ylabel('G')
ax.set_title(f"Gradient norm sequence")
ax.plot(range(1, n+1), G, 'k')

plt.tight_layout()

fname = "lamb_lbfgs_seq.png"
plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
print(f"Saved {fname}")
plt.close()
plt.close('all')