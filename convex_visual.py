import numpy as np
import csv
from scipy.interpolate import UnivariateSpline, splrep, BSpline, splev
from scipy.ndimage import gaussian_filter1d
from find_schedule import find_schedule
import os
import torch
import glob

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

compute_schedules = True
show_log_schedules = False
gnorm_division = True

# How many points to subsample down to for the plots
nvals = 1000

# Smoothing amounts hand tuned for plotting
sigma1 = 0.0002
sigma2 = 0.1

fnames = glob.glob("norm_sequences/convex/*.csv")

nrows = len(fnames)

fig = plt.figure()
if show_log_schedules:
    ncols = 4
else:
    ncols = 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, nrows*0.75))

for i, fname in enumerate(fnames):
    gnorms = []
    print(f"Reading {fname}")
    body = False
    with open(fname) as f:
        reader = csv.reader(f)
        for row in reader:
            if body:
                gnorms.append(float(row[1]))
            body = True

    gnorms = np.array(gnorms)
    gnorms = gnorms[~np.isinf(gnorms)]

    #print(gnorms)
    #import pdb; pdb.set_trace()
    glen = len(gnorms)

    if glen < nvals:
        gnorms = np.interp(
            np.linspace(0, 1.0, nvals),
            np.linspace(0, 1.0, glen), 
            gnorms)
        glen = len(gnorms)


    print("filtering")
    gnorms_filtered1 = torch.tensor(gaussian_filter1d(gnorms, sigma=sigma1*glen, mode="nearest"))
    gnorms_filtered2 = torch.tensor(gaussian_filter1d(gnorms, sigma=sigma2*glen, mode="nearest"))

    gnorms_short = np.interp(
        np.linspace(0, 1.0, nvals),
        np.linspace(0, 1.0, glen), 
        gnorms_filtered1)

    gnorms_smoothed = np.interp(
        np.linspace(0, 1.0, nvals),
        np.linspace(0, 1.0, glen), 
        gnorms_filtered2)
    
    #print(gnorms_short)

    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    x = np.linspace(0, 100.0, nvals)

    name = os.path.basename(fname)
    print(f"Plotting {name}")

    ax = axs[i, 0]
    ax.xaxis.set_major_formatter(xticks)
    ax.plot(x, gnorms_short, 'k')
    ax.text(0.5, 0.7, name, ha="center", verticalalignment='center', 
            color='k', transform = ax.transAxes)
    if i == 0:
        ax.set_title(f"Gradient Norms")
    #ax.set_xlabel('k')
    #ax.set_ylabel('lamb')

    ax = axs[i, 1]
    ax.xaxis.set_major_formatter(xticks)
    ax.plot(x, gnorms_smoothed, 'k')
    if i == 0:
        ax.set_title(f"Smoothed Gradient Norms")

    if compute_schedules:
        print("Optimizing schedule")
        sched = find_schedule(torch.tensor(gnorms_smoothed))

        sched = sched[:-3]
        xsched = np.linspace(0, 100.0, len(sched))

        if gnorm_division:
            gnorms_div = np.interp(
                np.linspace(0, 1.0, len(sched)),
                np.linspace(0, 1.0, glen), 
                gnorms_filtered2)
            sched = sched/gnorms_div
            sched = sched/np.max(sched[:len(sched)//2])

        # Save schedule
        out_name = fname.replace(".csv", ".sched")
        with open(out_name, 'w') as f:
            for idx in range(len(sched)):
                f.write(f"{sched[idx]}\n")

        print(f"Wrote {out_name}")

        ax = axs[i, 2]
        ax.xaxis.set_major_formatter(xticks)
        ax.plot(xsched, sched, 'k')
        if i == 0:
            ax.set_title(f"Refined Schedule")

        if show_log_schedules:
            ax = axs[i, 3]
            ax.xaxis.set_major_formatter(xticks)
            ax.plot(xsched, sched, 'k')
            ax.set_yscale('log')
            if i == 0:
                ax.set_title(f"Refined Schedule (Log Axis)")


    plt.tight_layout()

fname = f"convex_grids_{nvals}"

if gnorm_division:
    fname += "_div"

#plt.savefig(fname + ".png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.savefig(fname + ".pdf", bbox_inches='tight', pad_inches=0, dpi=300)
print(f"Saved {fname}")
plt.close()
plt.close('all')