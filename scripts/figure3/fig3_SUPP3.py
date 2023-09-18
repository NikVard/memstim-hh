#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
from pathlib import Path

import json

import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.animation as animation
from scipy import signal as sig
from tensorpac.utils import PSD
from matplotlib import colors
from matplotlib import ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *
from src.figure_plots_parameters import *

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


# main program
if __name__ == "__main__":
    import argparse
    import parameters

    parser = argparse.ArgumentParser(description='Generate figure 2 - supplementary 1 from the paper')

    parser.add_argument('-id', '--input-dir',
                        type=str,
                        default=os.path.join('results', 'analysis', 'default'),
                        help='Input directory (relative to base directory)'
                        )

    parser.add_argument('-op', '--order-parameter',
                        action='store_true',
                        default=False,
                        help='Set to plot order parameter instead of phase.')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig2_SUPP3',
                        help='Name of the output figure [w/o file extension].')

    args = parser.parse_args()

    # Normalize the path for later
    args.input_dir = os.path.normpath(args.input_dir)

    """ Load the configuration file for this simulation """
    print('[+] Loading parameters file...')
    filename = 'parameters_bak.json'
    try:
        params = parameters.load(os.path.join(args.input_dir, 'None', 'panel_A', filename))
    except Exception as e:
        print('[!]' + "Error code " + str(e.errno) + ": " + e.strerror)
        sys.exit(-1)

    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Timing
    second = 1.
    ms = 1e-3
    duration = params["simulation"]["duration"]
    dt = params["simulation"]["dt"]
    t_stim = params["stimulation"]["onset"]
    fs = int(1/dt)
    winsize_FR = 5*ms
    overlap_FR = 0.9
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)
    binnum = int(duration/winsize_FR)
    tv = np.arange(0., duration, dt)
    t_lims = [4000*ms, 6000*ms] # ms : x-axs limits
    t_lims_post = [4000*ms, 6000*ms]
    duration_post = t_lims_post[1] - t_lims_post[0]

    # Area names and sizes
    areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
    area_labels = ['EC', 'DG', 'CA3', 'CA1']
    N_tot = [[params["areas"][area_labels[0]]["E"]["N"], params["areas"][area_labels[0]]["I"]["N"]],
             [params["areas"][area_labels[1]]["E"]["N"], params["areas"][area_labels[1]]["I"]["N"]],
             [params["areas"][area_labels[2]]["E"]["N"], params["areas"][area_labels[2]]["I"]["N"]],
             [params["areas"][area_labels[3]]["E"]["N"], params["areas"][area_labels[3]]["I"]["N"]]]

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'
    c_inh_RGB = np.array([191, 97, 106])/255
    c_exc_RGB = np.array([94, 129, 172])/255

    # Raster downsampling
    N_scaling = 100
    N_gap = 10

    # Max rhythm value
    rhythm_gain_val = 0.18
    rhythm_max_val = 0.2 # 0.22 nA

    # Firing rates plotting gap
    rates_gap = 10 # Hz

    # Set theta rhythm limits
    xlims_rhythm = [t for t in t_lims]
    ylims_rhythm = [-0.01, rhythm_max_val]

    # Set raster limits
    xlims_rasters = [t for t in t_lims]
    ylims_rasters = [0, 2*N_scaling+N_gap]

    # Set firing rate limits
    xlims_rates = [t for t in t_lims]
    ylims_rates = [-1, 400]

    # Text parameters
    sizebar_off = 50*ms # sizebar offset

    # Figure sizes
    fig_width = 7.5
    fig_height = 8.75
    dpi = 300

    # Panels
    panel_names = ["A.", "B.", "C."]
    data_dirs = []
    for idx, panel in enumerate(panel_names):
        current_dir = os.path.join(args.input_dir, 'None', f'panel_{panel[0]}')
        data_dirs.append(current_dir)


    """ Plot supplementary figures """
    print('[+] Generating the figure...')

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height))

    # Use gridspecs
    G_outer = fig.add_gridspec(3, 1, left=0.02, right=0.9, bottom=0.05, top=0.97,
                                            # height_ratios=(0.15, 0.2125, 0.2125, 0.2125, 0.2125),
                                            hspace=0.15)

    # Separate the panels
    G_A = GridSpecFromSubplotSpec(4, 1, height_ratios=(0.1,0.1,0.4,0.4), hspace=0.5, subplot_spec=G_outer[0])
    G_B = GridSpecFromSubplotSpec(4, 1, height_ratios=(0.1,0.1,0.4,0.4), hspace=0.5, subplot_spec=G_outer[1])
    G_C = GridSpecFromSubplotSpec(4, 1, height_ratios=(0.1,0.1,0.4,0.4), hspace=0.5, subplot_spec=G_outer[2])
    G_rhythm = []
    G_phase = []
    G_rasters = []
    G_FRs = []
    
    G_rhythm.append(G_A[0])
    G_rhythm.append(G_B[0])
    G_rhythm.append(G_C[0])
    
    G_phase.append(G_A[1])
    G_phase.append(G_B[1])
    G_phase.append(G_C[1])

    G_rasters.append(G_A[2])
    G_rasters.append(G_B[2])
    G_rasters.append(G_C[2])

    G_FRs.append(G_A[3])
    G_FRs.append(G_B[3])
    G_FRs.append(G_C[3])

    # Fix tight layout
    G_outer.tight_layout(fig)


    # Organize axes
    #------------------------
    axs_rhythm = []
    axs_phase = []
    axs_rasters = []
    axs_FRs = []

    for idx, Gs in enumerate(zip(G_rhythm, G_phase, G_rasters, G_FRs)):
        # Rhythm
        # ------------------------
        print('[>] Input rhythms')
        ax_rhythm = fig.add_subplot(Gs[0])
        axs_rhythm.append(ax_rhythm)

        # Set the title
        ax_rhythm.set_title('Theta Rhythm', fontsize=fsize_titles)

        # Set the x-y limits
        ax_rhythm.set_xlim(xlims_rhythm)
        ax_rhythm.set_ylim(ylims_rhythm)

        # Hide x-y axes
        ax_rhythm.xaxis.set_visible(False)
        ax_rhythm.yaxis.set_visible(False)

        # Hide some spines
        ax_rhythm.spines['top'].set_visible(False)
        ax_rhythm.spines['bottom'].set_visible(False)
        ax_rhythm.spines['left'].set_visible(False)
        ax_rhythm.spines['right'].set_visible(False)

        # Order Parameter / Phase
        # ------------------------
        if args.order_parameter:
            print('[>] Order Parameter')
            ax_common = fig.add_subplot(Gs[1], sharex=ax_rhythm, sharey=ax_rhythm)

            # Set the title
            ax_common.set_title('Order Parameter', fontsize=fsize_titles)

        else:
            print('[>] Phase')
            ax_common = fig.add_subplot(Gs[1], sharex=ax_rhythm)
            ylims_common = [-0.1, 2*np.pi+0.1]

            # Set the x-y limits
            ax_common.set_ylim(ylims_common)

            # Set the title
            ax_common.set_title('Phase', fontsize=fsize_titles)
        axs_phase.append(ax_common)

        # Hide x-y axes
        ax_common.xaxis.set_visible(False)
        ax_common.yaxis.set_visible(False)

        # Hide some spines
        ax_common.spines['top'].set_visible(False)
        ax_common.spines['bottom'].set_visible(False)
        ax_common.spines['left'].set_visible(False)
        ax_common.spines['right'].set_visible(False)


        # Rasters
        # ------------------------
        print('[>] Raster')

        # Create the axes and append them to the list
        ax_raster = fig.add_subplot(Gs[2], sharex=ax_rhythm)
        axs_rasters.append(ax_raster)

        # Set limits
        ax_raster.set_xlim(xlims_rasters)
        ax_raster.set_ylim(ylims_rasters)

        # Titles
        ax_raster.set_title('CA1 Activity', fontsize=fsize_titles)

        # Axes
        ax_raster.xaxis.set_visible(False)
        # ax_raster.yaxis.set_visible(False)

        # Spines
        ax_raster.spines['top'].set_visible(False)
        ax_raster.spines['bottom'].set_visible(False)
        ax_raster.spines['left'].set_visible(False)
        ax_raster.spines['right'].set_visible(False)

        # Fix the ytick locators
        ax_raster.yaxis.set_major_locator(ticker.NullLocator())
        ax_raster.yaxis.set_minor_locator(ticker.NullLocator())

        # Remove tick labels
        # ax_raster.xaxis.set_ticklabels([])
        ax_raster.yaxis.set_ticklabels([])


        # Firing Rates
        # ------------------------
        print('[>] Firing Rates')
        ax_rate = fig.add_subplot(Gs[3], sharex=ax_rhythm)
        axs_FRs.append(ax_rate)

        # Set the x-y limits
        ax_rate.set_ylim(ylims_rates)

        # Axes
        ax_rate.xaxis.set_visible(False)
        ax_rate.yaxis.set_visible(False)

        # Spines
        ax_rate.spines['top'].set_visible(False)
        ax_rate.spines['bottom'].set_visible(False)
        ax_rate.spines['left'].set_visible(False)
        ax_rate.spines['right'].set_visible(False)

        # Fix the ytick locators
        ax_rate.yaxis.set_major_locator(ticker.NullLocator())
        ax_rate.yaxis.set_minor_locator(ticker.NullLocator())

        # Remove tick labels
        # ax_rate.xaxis.set_ticklabels([])
        ax_rate.yaxis.set_ticklabels([])

    # Iterate over areas
    print('[!] Actually plotting stuff...')


    # =====================
    # Plot the input rhythm
    # =====================

    print('[+] Plotting rhythm...')
    eq_str = r"$Z(\theta)=-sin(\theta - (\theta_{peak} "
    offsets = ['+ pi/2', '- pi/2', '+ pi']
    for input_dir,ax,offset in zip(data_dirs, axs_rhythm, offsets):
        rhythm = np.loadtxt(os.path.join(input_dir, 'data', 'order_param_mon_rhythm.txt'))
        # rhythm /= rhythm_gain_val
        ax.plot(tv, rhythm, ls='-', c='k', linewidth=1., rasterized=False, zorder=1)

        # find the oscillation frequency in the post-stim window using TensorPAC toolbox
        idx_window = np.logical_and(np.around(tv,4)>=t_lims_post[0], np.around(tv,4)<t_lims_post[1])
        rhythm_window = rhythm[idx_window]
        rhythm_PSD = PSD(rhythm_window[np.newaxis,:], fs)

        # peak indicates most prominent] oscillation
        idx_max_val_psd = np.argmax(rhythm_PSD.psd[0])
        fval1 = rhythm_PSD.freqs[idx_max_val_psd]

        # text frequency label
        ax.text(x=xlims_rhythm[0]+100*ms, y=rhythm_max_val, s=r"$f_\theta={0:.1f}$ Hz".format(fval1), fontsize=fsize_misc, ha='left', va='bottom', color='k', clip_on=False)
        ax.text(x=xlims_rhythm[1]-10*ms, y=rhythm_max_val, s=eq_str+f"{offset}" +r"))$", fontsize=fsize_misc, ha='right', va='bottom', color='k', clip_on=False)

    # add a sizebar for the y-axis
    add_sizebar(axs_rhythm[0], [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 0.2], 'black', ['0', '0.2 nA'], fsize=fsize_misc, rot=[0, 0], textx=[xlims_rhythm[1]+1.5*sizebar_off]*2, texty=[0, 0.2], ha='left', va='center')

    # ================================
    # Plot the phase / order parameter
    # ================================
    for input_dir,ax in zip(data_dirs, axs_phase):
        if args.order_parameter:
            print('[+] Plotting order parameter...')
            data = np.loadtxt(os.path.join(input_dir, 'data', 'order_param_mon_coherence.txt'))

            # asymptote
            ax.hlines(y=1., xmin=0., xmax=duration, color='k', ls='--', linewidth=0.5, zorder=11)

        else:
            print('[+] Plotting phase...')
            data = np.loadtxt(os.path.join(input_dir, 'data', 'order_param_mon_phase.txt'))

            # data = (data + np.pi) % (2 * np.pi)
            data += (1.*(data<0)*2*np.pi)

        # Data plotting
        ax.plot(np.arange(0.,duration,dt), data, ls='-', c='k', linewidth=1., rasterized=False, zorder=1)

    if args.order_parameter:
        # add a sizebar for the y-axis
        add_sizebar(axs_phase[0], [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 1.], 'black', ['0', '1'], fsize=fsize_misc, rot=[0, 0], textx=[xlims_rhythm[1]+1.5*sizebar_off]*2, texty=[0, 1], ha='left', va='center')
    else:
        # add a sizebar for the y-axis
        add_sizebar(axs_phase[0], [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 2*np.pi], 'black', ['0', '$2\pi$'], fsize=fsize_misc, rot=[0, 0], textx=[xlims_rhythm[1]+1.5*sizebar_off]*2, texty=[0, 2*np.pi], ha='left', va='center')


    # ==================
    # Plot Rasters + FRs
    # ==================
    for input_dir, ax_raster, ax_FR in zip(data_dirs, axs_rasters, axs_FRs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)

            # load t-i arrays for this area
            print('[+] Loading the spikes for CA1')
            i_exc = np.loadtxt(os.path.join(input_dir, 'data', 'spikes', 'CA1_pyCAN_spikemon_i.txt'))
            t_exc = np.loadtxt(os.path.join(input_dir, 'data', 'spikes', 'CA1_pyCAN_spikemon_t.txt'))
            i_inh = np.loadtxt(os.path.join(input_dir, 'data', 'spikes', 'CA1_inh_spikemon_i.txt'))
            t_inh = np.loadtxt(os.path.join(input_dir, 'data', 'spikes', 'CA1_inh_spikemon_t.txt'))

        i_exc = i_exc.astype(int)
        t_exc = t_exc*ms
        i_inh = i_inh.astype(int)
        t_inh = t_inh*ms

        # sort based on index number (lower to higher)
        idx_sort_exc = np.argsort(i_exc)
        idx_sort_inh = np.argsort(i_inh)
        i_exc = i_exc[idx_sort_exc]
        t_exc = t_exc[idx_sort_exc]
        i_inh = i_inh[idx_sort_inh]
        t_inh = t_inh[idx_sort_inh]

        # set number of neurons
        N_exc = N_tot[3][0]
        N_inh = N_tot[3][1]

        # select some neurons randomly, subscaling
        exc_mixed = np.arange(0, N_exc+1, int(N_exc/N_scaling))
        inh_mixed = np.arange(0, N_inh+1, int(N_inh/N_scaling))

        idx_exc = np.in1d(i_exc, exc_mixed)
        idx_inh = np.in1d(i_inh, inh_mixed)

        i_exc_sub = i_exc[idx_exc]
        t_exc_sub = t_exc[idx_exc]
        i_inh_sub = i_inh[idx_inh]
        t_inh_sub = t_inh[idx_inh]

        # assign new neuron count numbers
        cnt = 0
        i_exc_sub_new = np.copy(i_exc_sub)
        for ii in exc_mixed:
            idx_tmp = np.where(i_exc_sub == ii)
            # print('changing ', ii, 'to ', cnt)
            i_exc_sub_new[idx_tmp] = cnt
            cnt += 1
        i_exc_sub = i_exc_sub_new

        # cnt = 0
        cnt += N_gap
        i_inh_sub_new = np.copy(i_inh_sub)
        for ii in inh_mixed:
            idx_tmp = np.where(i_inh_sub == ii)
            # print('changing ', ii, 'to ', cnt)
            i_inh_sub_new[idx_tmp] = cnt
            cnt += 1
        i_inh_sub = i_inh_sub_new

        # plot spikes
        print('[>] Plotting spikes...')

        # inhibitory
        ax_raster.scatter(t_inh_sub, i_inh_sub, s=1.25, linewidth=1., marker='.', c=c_inh, edgecolors='none', alpha=1., rasterized=True)

        # excitatory
        ax_raster.scatter(t_exc_sub, i_exc_sub, s=1.25, linewidth=1., marker='.', c=c_exc, edgecolors='none', alpha=1., rasterized=True)

        # mean FRs
        FR_inh_mean = (sum((t_inh>=t_lims_post[0]) & (t_inh<t_lims_post[1]))/duration_post)/N_inh
        FR_exc_mean = (sum((t_exc>=t_lims_post[0]) & (t_exc<t_lims_post[1]))/duration_post)/N_exc

        # add it as a text
        ax_raster.text(x=xlims_rates[1]+30*ms, y=1.75*N_scaling+N_gap, s=r'$\mu_I$: {0:.1f} Hz'.format(FR_inh_mean), fontsize=fsize_xylabels, ha='left', color=c_inh, clip_on=False)
        ax_raster.text(x=xlims_rates[1]+30*ms, y=N_scaling//2, s=r'$\mu_E$: {0:.1f} Hz'.format(FR_exc_mean), fontsize=fsize_xylabels, ha='left', color=c_exc, clip_on=False)

        # calculate firing rates
        print('[>] Computing firing rates...')
        tv_inh_FR, FR_inh, fs_FR2 = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
        tv_exc_FR, FR_exc, _ = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

        # Normalize the FRs
        FR_inh_norm = (FR_inh/winsize_FR)/N_inh
        FR_exc_norm = (FR_exc/winsize_FR)/N_exc

        # Plot the FRs
        print('[>] Plotting rates...')
        ax_FR.plot(tv_inh_FR, FR_inh_norm+FR_exc_norm.max()+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
        ax_FR.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)

        # Text mark inhibitory/excitatory in rasters
        ax_FR.text(x=xlims_rates[1]+30*ms, y=ylims_rates[0]+75+FR_exc_norm.max()+rates_gap, s='Inhibitory', fontsize=fsize_legends, ha='left', color=c_inh, clip_on=False)
        ax_FR.text(x=xlims_rates[1]+30*ms, y=ylims_rates[0]+25, s='Excitatory', fontsize=fsize_legends, ha='left', color=c_exc, clip_on=False)

    # # add a sizebar for the x-axis
    # xlims_sz = [xlims_rates[1]+sizebar_off, xlims_rates[1]+sizebar_off-200*ms]
    # add_sizebar(curr_ax_rates, xlims_sz, [-50, -50], 'black', '200 ms', fsize=fsize_misc, rot=0, textx=np.mean(xlims_sz), texty=-100, ha='center', va='top')

    # # add a sizebar for the y-axis
    # ylims_sz = [-50, 150]
    # add_sizebar(curr_ax_rates, [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], ylims_sz, 'black', '200 Hz', fsize=fsize_misc, rot=0, textx=xlims_rhythm[1]+1.5*sizebar_off, texty=np.mean(ylims_sz), ha='left', va='center')

    for ax,label in zip(axs_rhythm, panel_names):
        # Set the panel labels
        ax.set_title(label, loc='left', weight='bold', fontsize=fsize_panels)

    # Sizebar for x-axis
    xlims_sz = [t_lims[1] - 300*ms, t_lims[1]-50*ms]
    ylims_sz = [-15, 25]
    add_sizebar(axs_FRs[-1], xlims_sz, [-25]*2, 'black', '250 ms', fsize=fsize_xylabels, rot=0, textx=np.mean(xlims_sz), texty=ylims_sz[0]-40, ha='center', va='top')

    # Save and show the figure
    print('[+] Saving the figure...')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig2', args.figure_name + '.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig2', args.figure_name + '.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

    plt.show()

    # Exit - no errors
    sys.exit(0)
