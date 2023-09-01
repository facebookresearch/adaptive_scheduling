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
write_to_disk = True

# How many points to subsample down to for the plots
#nvals = 1000
nvals = 10_000
skip = 10 # Only plot every 10th point

# Smoothing amounts hand tuned for plotting
sigma1 = 0.0002
# Using 0.05 seems to smooth over roberta hump, but also extends warmups too much...
# Lets see
sigma2 = 0.03

fnames = glob.glob("norm_sequences/*.csv")

fnames = [f for f in fnames if "cifar" not in f]

name_map = {
    'dlrm.csv': 'DLRM',
    'gpt.csv': 'GPT',
    'iwslt14.csv': "IWSLT14",
    'roberta.csv': 'RoBERTa',
    "vit.csv": 'ViT',
    "mri.csv": 'MRI',
    'rcnn.csv': "RCNN",
    "imagenet.csv": "ImageNet",
}

gnorm_log_inset_map = {
    'dlrm.csv': 'upper right',
    'gpt.csv': 'upper right',
    'iwslt14.csv': "upper right",
    'roberta.csv': 'upper right',
    "vit.csv": 'upper right',
    "mri.csv": 'upper right',
    'rcnn.csv': "upper right",
    "imagenet.csv": "lower right",
}

sched_log_inset_map = {
    'dlrm.csv': 'upper right',
    'gpt.csv': 'upper right',
    'iwslt14.csv': "upper right",
    'roberta.csv': 'upper right',
    "vit.csv": 'upper right',
    "mri.csv": 'upper right',
    'rcnn.csv': "upper right",
    "imagenet.csv": 'upper right',
}

uses_adam = {
    'dlrm.csv': True,
    'gpt.csv': True,
    'iwslt14.csv': True,
    'roberta.csv': True,
    "vit.csv": True,
    "mri.csv": True,
    'rcnn.csv': False,
    "imagenet.csv": False,
}

warmup_perc = {
    'dlrm.csv': 0.08,
    'gpt.csv': 0.24,
    'iwslt14.csv': 0.1,
    'roberta.csv': 0.2,
    "vit.csv": 0.12,
    "mri.csv": 0.12,
    'rcnn.csv': 0.07,
    "imagenet.csv": 0.065,
}

poly_exp = {
    'dlrm.csv': 1.4,
    'gpt.csv': 1.2,
    'iwslt14.csv': 1.3,
    'roberta.csv': 1.0,
    "vit.csv": 1.5,
    "mri.csv": 0.5,
    'rcnn.csv': 2.0,
    "imagenet.csv": 2.4,
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
    #gnorms_filtered2 = median_filter(gnorms, size=1201, mode='nearest')
    pad = 1000
    gnorms_filtered2 = median_filter(np.pad(gnorms, (0, pad), mode='reflect'), size=1201, mode='nearest')[:-pad]

    #gnorms_filtered2 = torch.tensor(gaussian_filter1d(gnorms, sigma=sigma2*glen, mode="nearest"))

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
    ax.plot(x[::skip], gnorms_short[::skip], 'k')

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
    logaxis.plot(x[::skip], gnorms_short[::skip], 'k')
    logaxis.set_yscale('log')
    logaxis.patch.set_alpha(0.6)
    logaxis.get_xaxis().set_visible(False)
    #ax.set_xlabel('k')
    #ax.set_ylabel('lamb')

    ax = axs[i, 1]
    ax.xaxis.set_major_formatter(xticks)
    ax.plot(x[::skip], gnorms_smoothed[::skip], 'k')
    if i == 0:
        ax.set_title(f"Smoothed Gradient Norms")

    if compute_schedules:
        print("Optimizing schedule")
        sched = find_closed_form_schedule(gnorms_smoothed)

        sched = sched[:-3]
        xsched = np.linspace(0, 100.0, len(sched))
        sched_nognorm = sched/np.max(sched[:len(sched)//2])

        if write_to_disk:
            out_name = fname.replace(".csv", ".sched_nognorm_median")
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
        if write_to_disk:
            out_name = fname.replace(".csv", ".sched_median")
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
        ax.plot(xsched[::skip], sched[::skip], 'k',  linewidth=0.4)

        # sched_all = find_closed_form_schedule(gnorms)
        # sched_all = sched_all[:-3]*gnorms[:-3]
        # sched_all = sched_all/np.max(sched_all[:len(sched_all)//2])

        # xsched_all = np.linspace(0, 100.0, len(sched_all))
        # ax.plot(xsched_all, sched_all, 'green',  linewidth=0.2)

        nsched = len(sched)
        warmup_steps = warmup_perc[basename] * nsched
        
        poly_sched = np.zeros(nsched)
        for j in range(nsched):
            if j < warmup_steps:
                poly_sched[j] = (j+1)/warmup_steps
            else:   
                poly_sched[j] = (1-(j-warmup_steps)/(nsched - warmup_steps))**poly_exp[basename]
        
        ax.plot(xsched[::skip], poly_sched[::skip], 'blue', alpha=0.35)

        # acum = 0.0
        # alt_sched2 = find_closed_form_schedule(gnorms_smoothed)
        # alt_sched2 = max(sched) * alt_sched2/max(alt_sched2)
        # alt_sched2 = alt_sched2[:-3]

        # if uses_adam[basename]:
        #     alt_sched2 = alt_sched2*gnorms_mul
        #     alt_sched2 = alt_sched2/np.max(alt_sched2[:len(alt_sched2)//2])

        # ax.plot(xsched, alt_sched2, 'green', linewidth=0.6, alpha=0.5, label="Closed Form")

        #ax.legend()
        plt.tight_layout()

        if i == 0:
            ax.set_title(f"Refined Schedule")

        logaxis = inset_axes(ax,
                    width="30%", # width = 30% of parent_bbox
                    height=0.2, # height : 1 inch
                    loc=sched_log_inset_map[basename])
        logaxis.xaxis.set_major_formatter(xticks)
        logaxis.tick_params(axis='both', which='major', labelsize=2, pad=0)
        logaxis.tick_params(axis='both', which='minor', labelsize=2, pad=0)
        logaxis.plot(xsched[::skip], sched[::skip], 'k')
        logaxis.set_yscale('log')
        logaxis.patch.set_alpha(0.6)
        logaxis.get_xaxis().set_visible(False)

    plt.tight_layout()

fname = f"real_grids_{nvals}"

#plt.savefig(fname + ".png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.savefig(fname + ".pdf", bbox_inches='tight', pad_inches=0, dpi=300)
print(f"Saved {fname}")
plt.close()
plt.close('all')