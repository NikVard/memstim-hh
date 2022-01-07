#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

dt = .1 # [sec]

# make a time vector (200 ms, one period)
tv = np.linspace(0,250,2501)
PRC_all = []
xvals_all = []

# Open-loop results directory| no stim
dir_ol = "../results/None/OL/"
phase_ol = np.loadtxt(os.path.join(dir_ol, 'data', 'order_param_mon_phase.txt'))

# Closed-loop results directory | no stim
dir_cl = "../results/None/CL/"
phase_cl = np.loadtxt(os.path.join(dir_cl, 'data', 'order_param_mon_phase.txt'))

# Results directories
dirs = []
dirs.append("../results/5_nA")
dirs.append("../results/10_nA")
dirs.append("../results/12_nA")
dirs.append("../results/15_nA")
dirs.append("../results/20_nA")
dirs.append("../results/25_nA")

# set default phase to compare against
phase_def = phase_cl

# iterate over stimulation amplitudes
for root in dirs:

    # initialization of current stim amplitude lists
    PRC = []
    xvals = []

    # iterate over saved PRC results
    for item in os.listdir(root):

        # is this a simulation directory
        if os.path.isdir(os.path.join(root, item)):
            # get the stimulation onset [ms]
            tmp = item.split("_")
            t_on = float(tmp[1])

            # find index of phase reset (based on stim_on)
            idx = int(t_on/dt)

            # load the data for the current simulation
            frate = np.loadtxt(os.path.join(root, item, 'data', 's2r_mon_drive.txt'))
            phase = np.loadtxt(os.path.join(root, item, 'data', 'order_param_mon_phase.txt'))

            # find index where input actually takes effect
            #idx2 = np.argmax(frate)

            # calculate d_phi
            #d_phi = phase[1:] - phase[:-1]
            #d_phi = np.mean(phase[idx-10:idx]) - np.mean(phase[idx:idx+10])
            #d_phase = phase[1:] - phase[:-1]
            d_phi = phase[idx+25] - phase_def[idx+25]

            # phase difference @ time of actual phase reset
            #PRC[int(idx2-10000)] = d_phi[idx2]
            # PRC.append(d_phi[idx2])
            # xvals.append(phase[idx2])
            PRC.append(d_phi)
            xvals.append(phase[idx+25])

    # Get sorting indices
    idxs = np.argsort(xvals)

    # Sort the data points
    xvals = [xvals[i] for i in idxs]
    PRC = [PRC[i] for i in idxs]

    # add to list
    PRC_all.append(PRC)
    xvals_all.append(xvals)


# Plotting
# ------------------------------------------------------------------------------
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
fig.set_figheight(12)
fig.set_figwidth(16)

# 5-15nA
axs.plot(xvals_all[0], PRC_all[0], c='C0', ls='--', marker='^', markersize=11, label=r'5nA', zorder=10)
axs.plot(xvals_all[1], PRC_all[1], c='C1', ls='--', marker='X', markersize=11, label=r'10nA', zorder=11)
axs.plot(xvals_all[2], PRC_all[2], c='C2', ls='--', marker='D', markersize=11, label=r'12nA', zorder=12)
axs.plot(xvals_all[3], PRC_all[3], c='C3', ls='--', marker='H', markersize=11, label=r'15nA', zorder=13)
axs.plot(xvals_all[4], PRC_all[4], c='C4', ls='--', marker='*', markersize=11, label=r'20nA', zorder=14)
axs.plot(xvals_all[5], PRC_all[5], c='C5', ls='--', marker='o', markersize=11, label=r'25nA', zorder=15)

# Horizontal line @ y=0
axs.hlines(y=0., xmin=-2*np.pi, xmax=2*np.pi, linestyle=':', colors='k', linewidth=2., zorder=2)

# Vertical line @ x=0
axs.vlines(x=0., ymin=-1., ymax=1., linestyle=':', colors='k', linewidth=2., zorder=2)

# Fill positive (y>0) and negative (y<0)
axs.fill_between(x=[-4,4], y1=[1., 1.], color='green', alpha=0.1)
axs.fill_between(x=[-4,4], y1=[-1., -1.], color='red', alpha=0.1)

# Add text
boxp = dict(boxstyle='square', alpha=0.75, facecolor='white', edgecolor='none')
axs.text(x=-3, y=0.28, color='black', s='ADV', fontsize=11, verticalalignment='top', horizontalalignment='left', bbox=boxp, zorder=20)
axs.text(x=-3, y=-0.28, color='black', s='DEL', bbox=boxp, zorder=21)

# Limits
axs.set_xlim([-3.3, 3.3])
axs.set_ylim([-0.3, 0.3])

# Ticks
def format_func(x, pos):
    # find number of multiples of pi/2
    N = int(np.round(2 * x / np.pi))
    if N == -2:
        return r"-$\pi$"
    elif N == -1:
        return r"-$\pi/2$"
    elif N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

axs.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
axs.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
axs.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

# Labels
axs.set_xlabel('Stim. Phase [rad]')
axs.set_ylabel(r'PRC - $\Delta\phi$ [rad]')

# Title
axs.set_title("Phase Response Curve (PRC)")

# Grids
axs.grid()

# Legend
axs.legend()


# Show the figure
plt.show()
