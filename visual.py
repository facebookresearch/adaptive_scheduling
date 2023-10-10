import numpy as np
import csv
from scipy.interpolate import UnivariateSpline, splrep, BSpline, splev
from scipy.ndimage import gaussian_filter1d, median_filter
from find_schedule import find_schedule, find_closed_form_schedule
import os
import torch
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.titlesize'] = 6
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
write_to_disk = True

# How many points to subsample down to for the plots
nvals = 10_000
skip = 10 # Only plot every 10th point

main_line_width = 1

# Smoothing amount for left plot (unreadable without some smoothing)
sigma1 = 0.0002
# Smoothing for scheduling (middle plot)
sigma2 = 0.1

fnames = glob.glob("norm_sequences_l1/*.csv")

name_map = {
    "imagenet": "ImageNet",
    'iwslt14': "IWSLT14",
    'gpt': 'GPT',
    'roberta': 'RoBERTa',
    'dlrm': 'DLRM',
    "mri": 'MRI',
    "vit": 'ViT',
    'rcnn': "RCNN",
}

gnorm_log_inset_map = {
    'dlrm': 'upper right',
    'gpt': 'upper right',
    'iwslt14': "lower right",
    'roberta': 'upper right',
    "vit": 'upper right',
    "mri": 'upper right',
    'rcnn': "lower right",
    "imagenet": "lower right",
}

sched_log_inset_map = {
    'dlrm': 'upper right',
    'gpt': 'upper right',
    'iwslt14': "upper right",
    'roberta': 'upper right',
    "vit": 'upper right',
    "mri": 'upper right',
    'rcnn': "upper right",
    "imagenet": 'upper right',
}

warmup_perc = {
    'dlrm': 0.2,
    'gpt': 0.28,
    'iwslt14': 0.08,
    'roberta': 0.45,
    "vit": 0.07,
    "mri": 0.4,
    'rcnn': 0.0,
    "imagenet": 0.01,
}

poly_exp = {
    'dlrm': 0.9,
    'gpt': 1.0,
    'iwslt14': None,
    'roberta': 0.8,
    "vit": None,
    "mri": 0.9,
    'rcnn': 1.5,
    "imagenet": 3.0,
}

nrows = 8

fig = plt.figure()
ncols = 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, nrows*0.75))

runs = {}
for fname in fnames:
    full_basename = os.path.basename(fname)
    basename = full_basename.split("_")[0]
    runs[basename] = fname

runs_ordered = []

for name in name_map:
    runs_ordered.append(runs[name])

for i, fname in enumerate(runs_ordered):
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

    # Clip tail value to handle data issues from wandb exports
    gnorms = gnorms[2:-2]
    glen = len(gnorms)

    if glen < nvals:
        gnorms = np.interp(
            np.linspace(0, 1.0, nvals),
            np.linspace(0, 1.0, glen), 
            gnorms)
        glen = len(gnorms)


    print("filtering")
    gnorms_filtered1 = torch.tensor(gaussian_filter1d(gnorms, sigma=sigma1*glen, mode="nearest"))
    filter_width = 2*(int(sigma2*nvals)//2) + 1
    pad = 2*filter_width
    gnorms_filtered2 = median_filter(np.pad(gnorms, (0, pad), mode='reflect'), size=filter_width, mode='nearest')[:-pad]

    gnorms_short = np.interp(
        np.linspace(0, 1.0, nvals),
        np.linspace(0, 1.0, glen), 
        gnorms_filtered1)

    gnorms_smoothed = np.interp(
        np.linspace(0, 1.0, nvals),
        np.linspace(0, 1.0, glen), 
        gnorms_filtered2)

    fmt = '%.0f%%' # Percentage format for ticks, i.e. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    x = np.linspace(0, 100.0, nvals)

    full_basename = os.path.basename(fname)
    basename = full_basename.split("_")[0]
    name = name_map.get(basename, basename)
    print(f"Plotting {name}")

    ax = axs[i, 0]
    ax.xaxis.set_major_formatter(xticks)
    ax.plot(x[::skip], gnorms_short[::skip], linewidth=main_line_width, color='orange')

    text_loc = 0.8

    ax.text(0.3, text_loc, name, ha="center", verticalalignment='center', 
            color='k', transform = ax.transAxes)
    if i == 0:
        ax.set_title(f"Gradient Norms", fontweight='bold')

    logaxis = inset_axes(ax,
                    width="30%",
                    height=0.2,
                    loc=gnorm_log_inset_map[basename])
    logaxis.tick_params(axis='both', which='major', labelsize=2, pad=0)
    logaxis.tick_params(axis='both', which='minor', labelsize=2, pad=0)
    logaxis.plot(x[::skip], gnorms_short[::skip], 'k')
    logaxis.set_yscale('log')
    logaxis.patch.set_alpha(0.6)
    logaxis.get_xaxis().set_visible(False)

    ax = axs[i, 1]
    ax.xaxis.set_major_formatter(xticks)
    ax.plot(x[::skip], gnorms_smoothed[::skip], linewidth=main_line_width, color='orange')
    if i == 0:
        ax.set_title(f"Smoothed Gradient Norms", fontweight='bold')

    if compute_schedules:
        print("Optimizing schedule")

        if "l1" in full_basename:
            sched = find_closed_form_schedule(gnorms_smoothed, weights=gnorms_smoothed**-1)
            print("L1 version")
        else:
            sched = find_closed_form_schedule(gnorms_smoothed, weights=gnorms_smoothed**-2)
            print("^-2 version")
        
        xsched = np.linspace(0, 100.0, len(sched))
        sched = sched/np.max(sched)

        if write_to_disk:
            out_name = fname.replace(".csv", ".sched_median")
            with open(out_name, 'w') as f:
                for idx in range(len(sched)):
                    f.write(f"{sched[idx]}\n")
            print(f"Wrote {out_name}")

        ax = axs[i, 2]
        ax.xaxis.set_major_formatter(xticks)
        ax.plot(xsched[::skip], sched[::skip], linewidth=main_line_width, color='orange')

        nsched = len(sched)
        warmup_steps = warmup_perc[basename] * nsched
        
        if poly_exp[basename] is not None:
            poly_sched = np.zeros(nsched)
            for j in range(nsched):
                if j < warmup_steps:
                    poly_sched[j] = (j+1)/warmup_steps
                else:   
                    poly_sched[j] = (1-(j-warmup_steps)/(nsched - warmup_steps))**poly_exp[basename]
            
            ax.plot(xsched[::skip], poly_sched[::skip], 'blue', alpha=0.35)

        plt.tight_layout()

        if i == 0:
            ax.set_title(f"Refined Schedule", fontweight='bold')

        logaxis = inset_axes(ax,
                    width="30%",
                    height=0.2,
                    loc=sched_log_inset_map[basename])
        logaxis.xaxis.set_major_formatter(xticks)
        logaxis.tick_params(axis='both', which='major', labelsize=2, pad=0)
        logaxis.tick_params(axis='both', which='minor', labelsize=2, pad=0)
        logaxis.plot(xsched[::skip], sched[::skip], 'k')
        logaxis.set_yscale('log')
        logaxis.patch.set_alpha(0.6)
        logaxis.get_xaxis().set_visible(False)

    plt.tight_layout()

fname = f"visual_l1"

plt.savefig(fname + ".pdf", bbox_inches='tight', pad_inches=0, dpi=300)
print(f"Saved {fname}")
plt.close()
plt.close('all')