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
import torch
from scipy.interpolate import UnivariateSpline, splrep, BSpline, splev
from scipy.ndimage import gaussian_filter1d

from find_schedule import find_closed_form_schedule

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
mpl.rcParams['lines.linewidth'] = 0.4
mpl.rcParams['axes.linewidth'] = linewidth
mpl.rcParams['xtick.major.width'] = linewidth
mpl.rcParams['ytick.major.width'] = linewidth

n = 2000

x = np.linspace(0, 1, n)

grids = [
    np.interp(x=x, xp=[0, 1], fp=[1,1]),
    np.interp(x=x, xp=[0, 0.003, 0.02, 0.03, 0.1, 1], fp=[15, 6, 4, 2, 1, 1]),
    np.interp(x=x, xp=[0, 0.003, 0.02, 0.03, 0.1, 1], fp=[15, 6, 4, 2, 1, 2]),
    np.interp(x=x, xp=[0, 0.003, 0.02, 0.03, 0.1, 1], fp=[15, 6, 4, 2, 1, 0.7]),
]

## Smooth the schedules
sigma = 40.0
for i in range(len(grids)):
    grids[i] = gaussian_filter1d(grids[i], sigma=sigma, mode="nearest")

schedules = []
for G in grids:
    schedules.append(find_closed_form_schedule(torch.tensor(G)))

print(schedules)

ngrids = len(schedules)

fig = plt.figure()
fig, axs = plt.subplots(nrows=ngrids, ncols=2, figsize=(5, 3.5))

for i in range(len(schedules)):
    ax = axs[i, 0]
    xsched = np.linspace(0, 100.0, len(schedules[i]))
    plt.tight_layout()
    if i == 0:
        ax.set_title(f"Gradient Norm")
    ax.axhline(y=1.0, color='blue', linestyle='-.', linewidth=0.2)
    ax.plot(xsched, grids[i], 'k')
    if i != len(schedules) - 1:
        ax.get_xaxis().set_visible(False)
    ax.set_ylim((0, 10))
    ax.set_xlim((0, 100))
    plt.tight_layout()

    ax = axs[i, 1]
    plt.tight_layout()
    if i == 0:
        ax.set_title(f"Refined Schedule")
    ax.plot(xsched, schedules[i]/max(schedules[i]), 'k')
    if i != len(schedules) - 1:
        ax.get_xaxis().set_visible(False)
    ax.set_ylim((0, 1))
    ax.set_xlim((0, 100))
    plt.tight_layout()


fname = "synth_grids"
plt.savefig(fname + ".pdf", bbox_inches='tight', pad_inches=0, dpi=300)
print(f"Saved {fname}")
plt.close()
plt.close('all')