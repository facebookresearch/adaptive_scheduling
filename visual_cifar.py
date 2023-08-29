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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# How many points to subsample down to for the plots
#nvals = 1000
nvals = 2000

# Smoothing amounts hand tuned for plotting
sigma1 = 0.0002
# Using 0.05 seems to smooth over roberta hump, but also extends warmups too much...
# Lets see
sigma2 = 0.03

fnames = [
    "/private/home/adefazio/adaptive_scheduling/norm_sequences/cifar10.csv",
    "/private/home/adefazio/adaptive_scheduling/norm_sequences/cifar100.csv"
]


name_map = {
    'cifar10.csv': 'CIFAR10',
    'cifar100.csv': "CIFAR100",
}

gnorm_log_inset_map = {
    'cifar10.csv': 'lower left',
    'cifar100.csv': "lower left",
}

sched_log_inset_map = {
    'cifar10.csv': 'upper left',
    'cifar100.csv': "upper left",
}

uses_adam = {
    'cifar10.csv': False,
    'cifar100.csv': False,
}

nrows = len(name_map)

fig = plt.figure()
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

    # Clip tail value to handle data issues
    gnorms = gnorms[:-2]

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

    basename = os.path.basename(fname)
    name = name_map.get(basename, basename)
    if basename not in name_map:
        print(f"Skipping {name}")
        continue
    print(f"Plotting {name}")

    ax = axs[i, 0]
    ax.xaxis.set_major_formatter(xticks)
    ax.plot(x, gnorms_short, 'k')

    # Fix overlapping issue
    if basename == "imagenet.csv":
        text_loc = 0.65
    else:
        text_loc = 0.5

    ax.text(0.5, text_loc, name, ha="center", verticalalignment='center', 
            color='k', transform = ax.transAxes)
    if i == 0:
        ax.set_title(f"Gradient Norms")

    logaxis = inset_axes(ax,
                    width="30%", # width = 30% of parent_bbox
                    height=0.2, # height : 1 inch
                    loc=gnorm_log_inset_map[basename])
    logaxis.tick_params(axis='both', which='major', labelsize=2, pad=0)
    logaxis.tick_params(axis='both', which='minor', labelsize=2, pad=0)
    logaxis.plot(x, gnorms_short, 'k')
    logaxis.set_yscale('log')
    logaxis.patch.set_alpha(0.6)
    logaxis.get_xaxis().set_visible(False)
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
        sched_nognorm = sched/np.max(sched[:len(sched)//2])

        out_name = fname.replace(".csv", ".sched_nognorm")
        with open(out_name, 'w') as f:
            for idx in range(len(sched)):
                f.write(f"{sched_nognorm[idx]}\n")
        print(f"Wrote {out_name}")

        gnorms_mul = np.interp(
            np.linspace(0, 1.0, len(sched)),
            np.linspace(0, 1.0, glen), 
            gnorms_filtered2)
        sched_gnorm = sched*gnorms_mul
        sched_gnorm = sched_gnorm/np.max(sched_gnorm[:len(sched_gnorm)//2])

        # Save schedule
        out_name = fname.replace(".csv", ".sched")
        with open(out_name, 'w') as f:
            for idx in range(len(sched_gnorm)):
                f.write(f"{sched_gnorm[idx]}\n")
        print(f"Wrote {out_name}")

        if uses_adam[basename]:
            sched = sched_gnorm
        else:
            sched = sched_nognorm

        ax = axs[i, 2]
        ax.xaxis.set_major_formatter(xticks)
        ax.plot(xsched, sched, 'k')
        if i == 0:
            ax.set_title(f"Refined Schedule")

        logaxis = inset_axes(ax,
                    width="30%", # width = 30% of parent_bbox
                    height=0.2, # height : 1 inch
                    loc=sched_log_inset_map[basename])
        logaxis.xaxis.set_major_formatter(xticks)
        logaxis.tick_params(axis='both', which='major', labelsize=2, pad=0)
        logaxis.tick_params(axis='both', which='minor', labelsize=2, pad=0)
        logaxis.plot(xsched, sched, 'k')
        logaxis.set_yscale('log')
        logaxis.patch.set_alpha(0.6)
        logaxis.get_xaxis().set_visible(False)

    plt.tight_layout()

fname = f"cifar_grids_{nvals}"

#plt.savefig(fname + ".png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.savefig(fname + ".pdf", bbox_inches='tight', pad_inches=0, dpi=300)
print(f"Saved {fname}")
plt.close()
plt.close('all')