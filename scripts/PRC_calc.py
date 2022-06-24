#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42

dt = .1 # [msec]
fs = 10e3
delay = 25 # [samples]

# make a time vector (200 ms, one period)
tv = np.linspace(0,250,2501)
PRC_all = []
xvals_all = []

# Open-loop results directory| no stim
# dir_cl = os.path.join('..', 'results', 'None', 'OL')
# phase_ol = np.loadtxt(os.path.join(dir_ol, 'data', 'order_param_mon_phase.txt'))

# Closed-loop results directory | no stim
dir_cl = os.path.join('results_PRC_new', 'None')
dir_cl = os.path.join(dir_cl, os.listdir(dir_cl)[0])
phase_cl = np.loadtxt(os.path.join(dir_cl, 'data', 'order_param_mon_phase.txt'))

# Load the CL rhythm
rhythm_cl = np.loadtxt(os.path.join(dir_cl, 'data', 'order_param_mon_rhythm.txt'))

# Results directories
dirs = []
dirs.append(os.path.join('results_PRC_new', '2_nA'))
dirs.append(os.path.join('results_PRC_new', '4_nA'))
dirs.append(os.path.join('results_PRC_new', '10_nA'))
dirs.append(os.path.join('results_PRC_new', '20_nA'))
dirs.append(os.path.join('results_PRC_new', '40_nA'))
# dirs.append(os.path.join('results_PRC_new', '10_nA'))

# set default phase to compare against
phase_def = phase_cl

# stimulation points for rhythm plotting later
t_stim_arr = []

# iterate over stimulation amplitudes
for root in dirs:
    print(root)
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
            print(t_on)
            t_stim_arr.append(t_on)

            # find index of phase reset (based on stim_on)
            idx = int(t_on/dt)

            # load the data for the current simulation
            curr_path = os.path.join(root, item)
            curr_path = os.path.join(curr_path, os.listdir(curr_path)[0])

            frate = np.loadtxt(os.path.join(curr_path, 'data', 's2r_mon_drive.txt'))
            phase = np.loadtxt(os.path.join(curr_path, 'data', 'order_param_mon_phase.txt'))

            # calculate d_phi
            # d_phi = np.mean(phase[idx-10:idx]) - np.mean(phase[idx:idx+10])
            # d_phi = phase[idx+delay] - phase_def[idx+delay]
            d_phi = (phase[idx+delay] - phase_def[idx+delay])
            d_phi = (d_phi + np.pi) % (2*np.pi) - np.pi     # wrap between [-pi pi]

            PRC.append(d_phi)
            xvals.append(phase[idx])

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
axs.plot(xvals_all[0], PRC_all[0], c='C0', ls='--', marker='^', markersize=11, label=r'2nA', zorder=10)
axs.plot(xvals_all[1], PRC_all[1], c='C1', ls='--', marker='X', markersize=11, label=r'4nA', zorder=11)
axs.plot(xvals_all[2], PRC_all[2], c='C2', ls='--', marker='D', markersize=11, label=r'10nA', zorder=12)
axs.plot(xvals_all[3], PRC_all[3], c='C3', ls='--', marker='H', markersize=11, label=r'20nA', zorder=13)
axs.plot(xvals_all[4], PRC_all[4], c='C4', ls='--', marker='*', markersize=11, label=r'40nA', zorder=14)
# axs.plot(xvals_all[5], PRC_all[5], c='C5', ls='--', marker='o', markersize=11, label=r'25nA', zorder=15)

# Plot the theoretical PRC in the background
phase_vec = np.linspace(-np.pi, np.pi,512)
axs.plot(phase_vec, -np.sin(phase_vec), c='k', zorder=0, label=r'Theoretical')

# find the time indices for the peri-stim phase cycle
tmin = np.array(t_stim_arr).min()
tmax = np.array(t_stim_arr).max()
tmin_idx = int(tmin/1000*fs)
tmax_idx = int(tmax/1000*fs)

# phase / rhythm
phase_cut = phase_cl[tmin_idx:tmax_idx]
rhythm_cut = rhythm_cl[tmin_idx:tmax_idx]

# Get sorting indices
idxs = np.argsort(phase_cut)

# Plot the actual theta rhythm w.r.t. phase
axs.plot(phase_cut[idxs], rhythm_cut[idxs]/rhythm_cut.max(), c='r', zorder=0, label=r'Rhythm')


# Horizontal line @ y=0
axs.hlines(y=0., xmin=-2*np.pi, xmax=2*np.pi, linestyle=':', colors='k', linewidth=2., zorder=2)

# Vertical line @ x=0
axs.vlines(x=0., ymin=-4., ymax=4., linestyle=':', colors='k', linewidth=2., zorder=2)

# Fill positive (y>0) and negative (y<0)
axs.fill_between(x=[-4,4], y1=[4., 4.], color='green', alpha=0.1)
axs.fill_between(x=[-4,4], y1=[-4., -4.], color='red', alpha=0.1)

# Add text
boxp = dict(boxstyle='square', alpha=0.75, facecolor='white', edgecolor='none')
axs.text(x=-3.3, y=0.28, color='black', s='ADV', fontsize=11, verticalalignment='top', horizontalalignment='left', bbox=boxp, zorder=20)
axs.text(x=-3.3, y=-0.28, color='black', s='DEL', bbox=boxp, zorder=21)

# Limits
axs.set_xlim([-3.3, 3.3])
axs.set_ylim([-3.5, 3.5])
axs.set_ylim([-1.1, 1.1])

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

axs.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
axs.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
axs.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

# Labels
axs.set_xlabel('Stim. Phase [rad]')
axs.set_ylabel(r'PRC - $\Delta\phi$ [rad]')

# Title
axs.set_title("Phase Response Curve (PRC)")

# Grids
axs.grid()

# Legend
axs.legend()

# Save the figures
fig.savefig('figures/PRC.pdf', transparent=True, dpi=200, format='pdf')
fig.savefig('figures/PRC.png', transparent=True, dpi=200, format='png')

# Show the figure
plt.show()
