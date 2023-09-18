#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# OS stuff
import os
import sys
import warnings
from pathlib import Path

# Computational stuff
import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import font_manager as fm
from matplotlib import ticker

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *
from src.figure_plots_parameters import *

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42

def add_sizebar(ax, xlocs, ylocs, bcolor, text, textx, texty, fsize, rot, ha, va):
    """ Add a sizebar to the provided axis """
    ax.plot(xlocs, ylocs, ls='-', c=bcolor, linewidth=1., rasterized=False, clip_on=False)

    # add the text
    if type(text) is list:
        # bottom text
        ax.text(x=textx[0], y=texty[0], s=text[0], rotation=rot[0], fontsize=fsize, va=va, ha=ha, clip_on=False)

        # top text
        ax.text(x=textx[1], y=texty[1], s=text[1], rotation=rot[1], fontsize=fsize, va=va, ha=ha, clip_on=False)
    else:
        ax.text(x=textx, y=texty, s=text, rotation=rot, fontsize=fsize, va=va, ha=ha, clip_on=False)

# Main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate revision figure on gamma oscillations')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig_rev_gamma',
                        help='Name of the output figure [w/o file extension]')

    args = parser.parse_args()

    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Directories
    data_dir = os.path.join(parent_dir, 'jupyter', 'data', 'fig_gamma_small_net')
    spikes_dir = os.path.join(data_dir, 'spikes')

    # Timing
    second = 1
    ms = 1e-3
    duration = 5*second
    dt = 0.1*ms
    fs = int(1*second/dt)
    tv = np.linspace(0, duration, fs*duration)/ms
    winsize_FR = 5*ms

    # Area names and sizes
    areas = ['pyCAN', 'inh']
    N_E = 1000
    N_I = 100

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'

    # Figure sizes
    fig_width = 7.5
    fig_height = 8.0

    # What data do we want to load?
    connection = ['DC', 'CON_Amelie']
    ramp_E = 1
    ramp_I = 0
    noise_I_list = [10, 100, 200]
    noise_E_list = [100, 1000, 2000]

    # Ramping input - non-repeating
    dt_ramp = 1*ms
    fs_ramp = 1/dt_ramp
    ramp_min = 0.
    ramp_max = 1.
    ramp_peak_t = 5*second
    tv_ramp = np.arange(0, duration, dt_ramp)
    xvals = np.arange(0, (ramp_peak_t/second), dt_ramp /second)
    ramp = ramp_min + (ramp_max-ramp_min)/(ramp_peak_t/second) * xvals

    # Parameters for FRs
    window_size = 1*ms
    bin_num = int(duration/window_size)
    fs_FR = 1/window_size

    # Filenames
    fname = f"input_ramp_E_{bool(ramp_E)}_I_{bool(ramp_I)}_noise_E_{noise_E_list[2]:0.0f}uV_I_{noise_I_list[2]:0.0f}uV"

    # Load the spike data
    # E
    E_i = np.loadtxt(os.path.join(spikes_dir, f"{connection[0]}_{areas[0]}_{fname}_i.txt"))
    E_t = np.loadtxt(os.path.join(spikes_dir, f"{connection[0]}_{areas[0]}_{fname}_t.txt"))
    E_t *= ms

    # I
    I_i = np.loadtxt(os.path.join(spikes_dir, f"{connection[0]}_{areas[1]}_{fname}_i.txt"))
    I_t = np.loadtxt(os.path.join(spikes_dir, f"{connection[0]}_{areas[1]}_{fname}_t.txt"))
    I_t *= ms
    # Compute the FRs
    FR_E, tv_E = np.histogram(E_t, bins=bin_num, range=[0,duration])
    FR_I, tv_I = np.histogram(I_t, bins=bin_num, range=[0,duration])

    # Adjust the time vectors
    tv_E = tv_E[0:-1] + window_size/2
    tv_I = tv_I[0:-1] + window_size/2


    # Plotting here
    # -------------------------------------------------
    # Figure sizes
    fig_width = 7.5
    fig_height = 8.0

    # Limits
    xlims_time = [0, 2]
    xlims_low = [0.4, 0.5]
    xlims_mid = [1.1, 1.25]
    ylims_input = [0, 0.5]
    ylims_rasters = [0, N_E+N_I]
    ylims_FRs = []

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height), tight_layout=True)
    # fig, axs = plt.subplots(3,1, figsize=(8,8))

    G_outer = GridSpec(1, 3, #left=0.1, right=0.95, bottom=0.15, top=0.925,
               width_ratios=(0.6, 0.2, 0.2),
               # hspace=0.5, wspace=0.5,
               figure=fig)
    G_inner = G_outer[0].subgridspec(3, 1)
    G_low = G_outer[1].subgridspec(3,1)
    G_mid = G_outer[2].subgridspec(3,1)

    ax_input = fig.add_subplot(G_inner[0])
    ax_rasters = fig.add_subplot(G_inner[1], sharex=ax_input)
    ax_FRs = fig.add_subplot(G_inner[2], sharex=ax_input)
    ax_FRs.set_xlim(xlims_time)

    ax_input_low = fig.add_subplot(G_low[0])
    ax_rasters_low = fig.add_subplot(G_low[1], sharex=ax_input_low)
    ax_FRs_low = fig.add_subplot(G_low[2], sharex=ax_input_low)
    ax_FRs_low.set_xlim(xlims_low)
    
    ax_input_mid = fig.add_subplot(G_mid[0])
    ax_rasters_mid = fig.add_subplot(G_mid[1], sharex=ax_input_mid)
    ax_FRs_mid = fig.add_subplot(G_mid[2], sharex=ax_input_mid)
    ax_FRs_mid.set_xlim(xlims_mid)

    # Plot the data
    ax_input.plot(tv_ramp, ramp)
    ax_input.set_title('Input')
    ax_input_low.plot(tv_ramp, ramp)
    ax_input_mid.plot(tv_ramp, ramp)

    ax_rasters.plot(E_t, E_i, 'b.', markersize=1)
    ax_rasters.plot(I_t, I_i+N_E, 'r.', markersize=1)
    ax_rasters.set_title('Rasters')
    ax_rasters_low.plot(E_t, E_i, 'b.', markersize=1)
    ax_rasters_low.plot(I_t, I_i+N_E, 'r.', markersize=1)
    ax_rasters_mid.plot(E_t, E_i, 'b.', markersize=1)
    ax_rasters_mid.plot(I_t, I_i+N_E, 'r.', markersize=1)

    ax_FRs.plot(tv_E, (FR_E/N_E), 'b')
    ax_FRs.plot(tv_I, (FR_I/N_I)+1, 'r')
    ax_FRs.set_title('Normalized rates')
    ax_FRs.set_ylabel('Time [s]')
    ax_FRs_low.plot(tv_E, (FR_E/N_E), 'b')
    ax_FRs_low.plot(tv_I, (FR_I/N_I)+1, 'r')
    ax_FRs_mid.plot(tv_E, (FR_E/N_E), 'b')
    ax_FRs_mid.plot(tv_I, (FR_I/N_I)+1, 'r')

    # Set the xlims
    ax_input.set_xlim(xlims_time)
    # ax_rasters.set_xlim(xlims_time)
    # ax_FRs.set_xlim(xlims_time)

    # Show the figure
    G_outer.tight_layout(fig)
    plt.show()

    # Exit properly
    sys.exit(0)