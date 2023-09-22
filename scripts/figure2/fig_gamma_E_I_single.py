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
parent_dir = Path(script_dir).parent.parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *
from src.figure_plots_parameters import *

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42
mplb.rcParams['axes.titlesize'] = 11
mplb.rcParams['axes.labelsize'] = 9

# Arial font everywhere
# ILLUSTRATOR STUFF
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
})

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

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
    figures_dir = os.path.join(data_dir, 'figures')

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
    connections = ['DC', 'CON_Amelie']
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


    for conn in connections:
        for ramp_E in [0, 1]:
            for ramp_I in [0 ,1]:
                for noise_E in noise_E_list:
                    for noise_I in noise_I_list:
                        
                        # Filenames
                        fname = f"input_ramp_E_{bool(ramp_E)}_I_{bool(ramp_I)}_noise_E_{noise_E:0.0f}uV_I_{noise_I:0.0f}uV"

                        # Load the spike data
                        # E
                        try:
                            E_i = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[0]}_{fname}_i.txt"))
                        except OSError:
                            print('File not found for parameter set:')
                            print(f'\tConnectivity: {conn}')
                            print(f'\tramp_E: {ramp_E} | ramp_I {ramp_I}')
                            print(f'\tnoise_E: {noise_E} | noise_I {noise_I}')
                            print('Continuing...')
                            print('-'*16)
                            continue

                        E_t = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[0]}_{fname}_t.txt"))
                        E_t *= ms

                        # I
                        I_i = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[1]}_{fname}_i.txt"))
                        I_t = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[1]}_{fname}_t.txt"))
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
                        xlims_time = [0, 3]
                        ylims_input = [0, 0.6]
                        ylims_rasters = [0, N_E+N_I]
                        ylims_FRs = [-0.1, 2.1]

                        # Make a figure
                        fig = plt.figure(figsize=(fig_width,fig_height))

                        G_outer = GridSpec(3,1, #left=0.1, right=0.95, bottom=0.15, top=0.925,
                                   height_ratios=(0.2, 0.4, 0.4),
                                   # hspace=0.5, wspace=0.5,
                                   figure=fig)
                        ax_input = fig.add_subplot(G_outer[0])
                        ax_rasters = fig.add_subplot(G_outer[1], sharex=ax_input)
                        ax_FRs = fig.add_subplot(G_outer[2], sharex=ax_input)

                        # Plot the data
                        ax_input.plot(tv_ramp, ramp, color='k')
                        ax_input.set_ylim(ylims_input)

                        ax_rasters.plot(E_t, E_i, linestyle='None', color=c_exc, marker='.', markersize=1, rasterized=True)
                        ax_rasters.plot(I_t, I_i+N_E, linestyle='None', color=c_inh, marker='.', markersize=1, rasterized=True)

                        ax_FRs.plot(tv_E, (FR_E/N_E), color=c_exc, label='Excitatory')
                        ax_FRs.plot(tv_I, (FR_I/N_I)+1, color=c_inh, label='Inhibitory')


                        # Fixed tick locators
                        ax_input.set_xticks(np.arange(0, max(xlims_time)+0.1, 0.5))

                        # Set the xlims/ylims
                        ax_input.set_xlim(xlims_time)
                        ax_rasters.set_ylim(ylims_rasters)
                        ax_FRs.set_ylim(ylims_FRs)

                        # Proper fontsizes
                        ax_input.tick_params(axis='both', which='both', labelsize=fsize_ticks)
                        ax_rasters.tick_params(axis='both', which='both', labelsize=fsize_ticks)
                        ax_FRs.tick_params(axis='both', which='both', labelsize=fsize_ticks)

                        # Remove the box borders
                        ax_input.spines['top'].set_visible(False)
                        ax_input.spines['right'].set_visible(False)
                        ax_input.spines['bottom'].set_visible(False)
                        ax_input.spines['left'].set_visible(False)

                        ax_rasters.spines['top'].set_visible(False)
                        ax_rasters.spines['right'].set_visible(False)
                        ax_rasters.spines['bottom'].set_visible(False)
                        ax_rasters.spines['left'].set_visible(False)

                        ax_FRs.spines['top'].set_visible(False)
                        ax_FRs.spines['right'].set_visible(False)
                        ax_FRs.spines['bottom'].set_visible(False)
                        ax_FRs.spines['left'].set_visible(False)

                        # Remove ticks
                        ax_input.get_xaxis().set_ticks([])
                        ax_input.get_yaxis().set_ticks([])
                        ax_rasters.get_xaxis().set_ticks([])
                        ax_rasters.get_yaxis().set_ticks([])
                        ax_FRs.get_xaxis().set_ticks([])
                        ax_FRs.get_yaxis().set_ticks([])

                        # Titles
                        ax_input.set_title('Input', loc='center', fontsize=fsize_titles)
                        ax_rasters.set_title('Rasters', loc='center', fontsize=fsize_titles)
                        ax_FRs.set_title('Normalized rates', loc='center', fontsize=fsize_titles)

                        # Add sizebars
                        add_sizebar(ax_input, [xlims_time[1]+0.03, xlims_time[1]+0.03], [0, 0.1], 'black', ['0 nA', '0.1'], fsize=fsize_misc, rot=[0, 0], 
                                    textx=[xlims_time[1]+0.05]*2, texty=[0, 0.1],
                                    ha='left', va='center')

                        add_sizebar(ax_FRs, [xlims_time[1]-0.25, xlims_time[1]-0.], [-0.1, -0.1], 'black', '250ms', fsize=fsize_misc, rot=0, 
                                    textx=np.mean([xlims_time[1]-0.25, xlims_time[1]-0.]), texty=-0.13, 
                                    ha='center', va='top')

                        # Set the xlims
                        ax_input.set_xlim(xlims_time)
                        # ax_rasters.set_xlim(xlims_time)
                        # ax_FRs.set_xlim(xlims_time)

                        # Show the figure
                        G_outer.tight_layout(fig)
                        
                        # Save the figure
                        fig.savefig(os.path.join(figures_dir, f"{conn}_{areas[0]}_{fname}_rev_eLife.png"))
                        fig.savefig(os.path.join(figures_dir, f"{conn}_{areas[0]}_{fname}_rev_eLife.pdf"))

                        # Close the figure
                        plt.close()

    # Plotting done
    print('Done with generating figures.')

    # Exit properly
    sys.exit(0)