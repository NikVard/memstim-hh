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
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib import ticker

# TensorPAC
from tensorpac import Pac, PreferredPhase
from tensorpac.utils import PSD

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *
from src.figure_plots_parameters import *

# Set font to Arial -- is this working?
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

# Error print!
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Add sizebar to figure
def add_sizebar(ax, xlocs, ylocs, bcolor, text, textx, texty, fsize, rot, ha='center', va='center'):
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

    parser = argparse.ArgumentParser(description='Generate supplementary figures for fig4 from paper')

    parser.add_argument('-ra', '--rasters-all',
                        action='store_true',
                        default=False,
                        help='Set to plot all rasters instead of only for CA1.')

    parser.add_argument('-op', '--order-parameter',
                        action='store_true',
                        default=False,
                        help='Set to plot the order parameter instead of the phase.')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig4_supp',
                        help='Name of the output figure [w/o file extension].')

    args = parser.parse_args()


    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Area names and sizes
    areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
    area_labels = ['EC', 'DG', 'CA3', 'CA1']
    N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

    # Timing
    second = 1.
    ms = 1e-3
    dt = 0.1*ms
    fs = int(1/dt)

    # duration = 12.*second # change this if you change the dataset
    duration = 10*second
    tv = np.arange(0, duration, dt)

    winsize_FR = 5*ms
    overlap_FR = 0.9
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)

    t_stim = 1800*ms # change this if you change the dataset
    t_lims = [t_stim - 500*ms, t_stim+6500*ms] # ms : x-axs limits
    t_lims_adj = [t_stim + 500*ms, t_stim+5500*ms] # ms : calculate mean FRs in a 5-sec window
    t_lims_adj = [1000*ms, 4000*ms] # ms : calculate mean FRs in a 5-sec window
    duration_adj = t_lims_adj[1] - t_lims_adj[0]

    # For Fig2. supp.
    # t_stim = 10801.9*ms
    # t_lims = [8200*ms, 11200*ms] # ms : x-axs limits
    # t_lims_adj = [4000*ms, 9000*ms] # ms : calculate mean FRs in a 5-sec window
    # duration_adj = t_lims_adj[1] - t_lims_adj[0]

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'
    c_inh_RGB = np.array([191, 97, 106])/255
    c_exc_RGB = np.array([94, 129, 172])/255

    # Raster downsampling
    N_scaling = 100
    N_gap = 10

    # Firing rates plotting gap
    rates_gap = 10 # Hz

    # Firing Rate computation parameters
    window_size = 100*ms
    window_width = int(fs_FR*window_size) # samples
    overlap = 0.99 # percentage
    noverlap = int(overlap*window_width)

    # Set theta rhythm limits
    xlims_rhythm = [t for t in t_lims]
    ylims_rhythm = [-0.1, 1.1]

    # Set raster limits
    xlims_rasters = [t for t in t_lims]
    ylims_rasters = [0, 2*N_scaling+N_gap]

    # Set firing rate limits
    xlims_rates = [t for t in t_lims]
    ylims_rates = [-1, 500]

    # Text parameters
    sizebar_off = 100*ms # sizebar offset

    # Figure sizes
    fig_width = 7.5
    fig_height = 8.75
    dpi = 300

    # Data
    results_dir = os.path.join(parent_dir, 'results_cluster', 'results_fig4_ext', 'K_0.18', '8.0_nA', '0.00_1800.0_ms', '06-12-2022 15H42M51S') # change this for new dataset
    results_dir = os.path.join(parent_dir, 'results', 'analysis', 'current', 'desc3') # change this for new dataset
    data_dir = os.path.join(results_dir, 'data')
    spikes_dir = os.path.join(data_dir, 'spikes')

    """ Plot supplementary figures for fig4 of the paper - TODO: Add DOI"""
    print('[+] Generating the figure...')

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height))

    # Use gridspecs
    G_outer = fig.add_gridspec(3, 1, left=0.075, right=0.925, bottom=0.075, top=0.925,
                        wspace=0.05, hspace=0.2, height_ratios=(0.02, 0.02, 0.6))
    G_rhythm = GridSpecFromSubplotSpec(1, 1, hspace=0.1, subplot_spec=G_outer[0])
    G_order_param = G_phase = GridSpecFromSubplotSpec(1, 1, hspace=0.1, subplot_spec=G_outer[1])
    G_all_areas = GridSpecFromSubplotSpec(4, 1, hspace=0.5, subplot_spec=G_outer[2])
    G_EC = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_all_areas[0])
    G_DG = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_all_areas[1])
    G_CA3 = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_all_areas[2])
    G_CA1 = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_all_areas[3])

    G_outer.tight_layout(fig)


    # Organize axes
    #------------------------
    axs = []


    # Rhythm
    # ------------------------
    print('[>] Input rhythm')
    ax_rhythm = fig.add_subplot(G_rhythm[0])
    axs.append(ax_rhythm)

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
        ax_common = fig.add_subplot(G_order_param[0], sharex=ax_rhythm, sharey=ax_rhythm)

        # Set the title
        ax_common.set_title('Order Parameter', fontsize=fsize_titles)

    else:
        print('[>] Phase')
        ax_common = fig.add_subplot(G_phase[0], sharex=ax_rhythm)
        ylims_common = [-0.1, 2*np.pi+0.1]

        # Set the x-y limits
        ax_common.set_ylim(ylims_common)

        # Set the title
        ax_common.set_title('Phase', fontsize=fsize_titles)

    axs.append(ax_common)

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
    print('[>] Rasters')

    # Create the axes and append them to the list
    ax_rasters_0 = fig.add_subplot(G_EC[0], sharex=ax_rhythm)
    ax_rasters_1 = fig.add_subplot(G_DG[0], sharex=ax_rhythm, sharey=ax_rasters_0)
    ax_rasters_2 = fig.add_subplot(G_CA3[0], sharex=ax_rhythm, sharey=ax_rasters_0)
    ax_rasters_3 = fig.add_subplot(G_CA1[0], sharex=ax_rhythm, sharey=ax_rasters_0)
    axs.append([ax_rasters_0, ax_rasters_1, ax_rasters_2, ax_rasters_3])

    # Set the titles
    # ax_rasters_0.set_title('Network Activity Overview')

    # Set limits
    ax_rasters_0.set_xlim(xlims_rasters)
    ax_rasters_0.set_ylim(ylims_rasters)

    # Formatting the axes
    for ax, label in zip([ax for ax in axs[2]], area_labels):
        # Titles
        ax.set_title('{0} Activity'.format(label), fontsize=fsize_titles)

        # Labels
        ax.set_ylabel(label, rotation=0, labelpad=15., fontsize=fsize_xylabels)

        # Axes
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Fix the ytick locators
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        # Remove tick labels
        # ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])


    # Firing Rates
    # ------------------------
    print('[>] Firing Rates')
    ax_rates_0 = fig.add_subplot(G_EC[1], sharex=ax_rhythm)
    ax_rates_1 = fig.add_subplot(G_DG[1], sharex=ax_rhythm, sharey=ax_rates_0)
    ax_rates_2 = fig.add_subplot(G_CA3[1], sharex=ax_rhythm, sharey=ax_rates_0)
    ax_rates_3 = fig.add_subplot(G_CA1[1], sharex=ax_rhythm, sharey=ax_rates_0)
    axs.append([ax_rates_0, ax_rates_1, ax_rates_2, ax_rates_3])

    # Set the title
    # ax_rates_0.set_title('Firing Rates')

    # Set the x-y limits
    ax_rates_0.set_ylim(ylims_rates)

    # Formatting the axes
    for ax, label in zip([ax for ax in axs[3]], area_labels):
        # Labels
        ax.set_ylabel(label, rotation=0, labelpad=15., fontsize=fsize_xylabels)

        # Axes
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Fix the ytick locators
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        # Remove tick labels
        # ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    # Iterate over areas
    print('[!] Actually plotting stuff...')


    # =====================
    # Plot the input rhythm
    # =====================

    print('[+] Plotting rhythm...')

    rhythm = np.loadtxt(os.path.join(data_dir, 'order_param_mon_rhythm.txt'))
    ax_rhythm.plot(tv, rhythm/(np.max(rhythm)), ls='-', c='k', linewidth=1.2, rasterized=False, zorder=1)

    # vertical lines at x-points
    pks, _ = sig.find_peaks(rhythm, distance=int(50*ms*fs))

    # peak indices filtering (in adjusted window)
    pks_idx = np.logical_and(pks>=t_lims_adj[0]*fs, pks<=t_lims_adj[1]*fs)
    pks_new = pks[pks_idx]

    # calculate the frequency in the adjusted window
    fval0 = 1/(np.mean(pks_new[1:] - pks_new[0:-1])/fs) if len(pks_new)>1 else 1/(pks_new[0]/fs)

    # find the oscillation frequency in the post-stim window using TensorPAC toolbox
    idx_window = np.logical_and(tv>t_lims_adj[0], tv<=t_lims_adj[1])
    rhythm_window = rhythm[idx_window]
    rhythm_PSD = PSD(rhythm_window[np.newaxis,:], fs)

    # peak indicates most prominent] oscillation
    idx_max_val_psd = np.argmax(rhythm_PSD.psd[0])
    fval1 = rhythm_PSD.freqs[idx_max_val_psd]

    # select which fval to show
    # fval_fin = fval0 # custom peak-based mean calculation
    fval_fin = fval1 # tensorpac PSD argmax calculation

    # text frequency label
    ax_rhythm.text(x=xlims_rhythm[0]+150*ms, y=1.2, s=r"$f_\theta={0:.1f}$ Hz".format(fval_fin), fontsize=fsize_misc, ha='left', color='k', clip_on=False)

    # add a sizebar for the y-axis
    add_sizebar(ax_rhythm, [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 1.], 'black', ['0', '1'], fsize=fsize_misc, rot=[0, 0], textx=[xlims_rhythm[1]+sizebar_off+20*ms]*2, texty=[0, 1], ha='left', va='center')


    # ================================
    # Plot the phase / order parameter
    # ================================

    if args.order_parameter:
        print('[+] Plotting order parameter...')
        data = np.loadtxt(os.path.join(data_dir, 'order_param_mon_coherence.txt'))

        # asymptote
        ax_common.hlines(y=1., xmin=0., xmax=duration, color='k', ls='--', linewidth=0.5, zorder=11)

        # add a sizebar for the y-axis
        add_sizebar(ax_common, [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 1.], 'black', ['0', '1'], fsize=fsize_misc, rot=[0, 0], textx=[xlims_rhythm[1]+sizebar_off+20*ms]*2, texty=[0, 1], ha='left', va='center')

    else:
        print('[+] Plotting phase...')
        data = np.loadtxt(os.path.join(data_dir, 'order_param_mon_phase.txt'))

        # data = (data + np.pi) % (2 * np.pi)
        data += (1.*(data<0)*2*np.pi)

        # add a sizebar for the y-axis
        add_sizebar(ax_common, [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 2*np.pi], 'black', ['0', '$2\pi$'], fsize=fsize_misc, rot=[0, 0], textx=[xlims_rhythm[1]+sizebar_off+20*ms]*2, texty=[0, 2*np.pi], ha='left', va='center')

        # text with stimulation phase [deg/rad]
        # ax_common.scatter(x=t_stim, y=data[int(t_stim*fs)], s=12, marker='o', c='k')
        # ax_common.text(x=t_stim-75*ms, y=data[int(t_stim*fs)]+0.45, s=r"$\pi/2$", fontsize=fsize_misc, ha='left', color='k', clip_on=False)

    # Data plotting
    ax_common.plot(np.arange(0.,duration,dt), data, ls='-', c='k', linewidth=1.2, rasterized=False, zorder=1)


    # ==================
    # Plot Rasters + FRs
    # ==================

    for area_name, fname, N_area, curr_ax_rasters, curr_ax_rates in zip(area_labels, areas, N_tot, axs[2], axs[3]):
        # load t-i arrays for this area
        print('[+] Loading the spikes for area:', area_name)
        i_exc = np.loadtxt(os.path.join(spikes_dir, '{0}_spikemon_i.txt'.format(fname[0])))
        t_exc = np.loadtxt(os.path.join(spikes_dir, '{0}_spikemon_t.txt'.format(fname[0])))
        i_inh = np.loadtxt(os.path.join(spikes_dir, '{0}_spikemon_i.txt'.format(fname[1])))
        t_inh = np.loadtxt(os.path.join(spikes_dir, '{0}_spikemon_t.txt'.format(fname[1])))

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
        N_exc = N_area[0]
        N_inh = N_area[1]

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
        curr_ax_rasters.scatter(t_inh_sub, i_inh_sub, s=5., linewidth=1., marker='.', c=c_inh, edgecolors='none', alpha=1., rasterized=True)

        # excitatory
        curr_ax_rasters.scatter(t_exc_sub, i_exc_sub, s=5., linewidth=1., marker='.', c=c_exc, edgecolors='none', alpha=1., rasterized=True)

        # mean FRs
        FR_inh_mean = (sum((t_inh>=t_lims_adj[0]) & (t_inh<t_lims_adj[1]))/duration_adj)/N_inh
        FR_exc_mean = (sum((t_exc>=t_lims_adj[0]) & (t_exc<t_lims_adj[1]))/duration_adj)/N_exc

        # add it as a text
        curr_ax_rasters.text(x=xlims_rates[1]+50*ms, y=1.5*N_scaling+N_gap, s=r'$\mu_I$: {0:.1f} Hz'.format(FR_inh_mean), fontsize=fsize_xylabels, ha='left', color=c_inh, clip_on=False)
        curr_ax_rasters.text(x=xlims_rates[1]+50*ms, y=N_scaling//2, s=r'$\mu_E$: {0:.1f} Hz'.format(FR_exc_mean), fontsize=fsize_xylabels, ha='left', color=c_exc, clip_on=False)

        # plot rates
        print('[>] Plotting rates...')
        tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
        tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

        # SAVE THE NON-NORMALIZED FR SIGNALS
        np.savetxt(os.path.join(data_dir, 'FR_inh.txt'), FR_inh, fmt='%.8f')
        np.savetxt(os.path.join(data_dir, 'tv_inh.txt'), tv_inh_FR, fmt='%.8f')
        np.savetxt(os.path.join(data_dir, 'FR_exc.txt'), FR_exc, fmt='%.8f')
        np.savetxt(os.path.join(data_dir, 'tv_exc.txt'), tv_exc_FR, fmt='%.8f')

        # Normalize the FRs
        FR_inh_norm = (FR_inh/winsize_FR)/N_inh
        FR_exc_norm = (FR_exc/winsize_FR)/N_exc

        # Plot the FRs
        curr_ax_rates.plot(tv_inh_FR, FR_inh_norm+FR_exc_norm.max()+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
        curr_ax_rates.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)

    axs[3][0].text(x=xlims_rates[1]+50*ms, y=ylims_rates[0]+150+FR_exc_norm.max()+rates_gap, s='Inhibitory', fontsize=fsize_legends, ha='left', color=c_inh, clip_on=False)
    axs[3][0].text(x=xlims_rates[1]+50*ms, y=ylims_rates[0]-150, s='Excitatory', fontsize=fsize_legends, ha='left', color=c_exc, clip_on=False)

    # add a sizebar for the y-axis
    add_sizebar(curr_ax_rates, [xlims_rates[1]+sizebar_off, xlims_rates[1]+sizebar_off], [-50, 50], 'black', '100 Hz', fsize=fsize_misc, rot=0, textx=xlims_rates[1]+sizebar_off+20*ms, texty=0, ha='left', va='center')

    # add a sizebar for the x-axis
    xlims_sz = [xlims_rates[1]+sizebar_off, xlims_rates[1]+sizebar_off-200*ms]
    add_sizebar(curr_ax_rates, xlims_sz, [-50, -50], 'black', '200 ms', fsize=fsize_misc, rot=0, textx=np.mean(xlims_sz), texty=-75, ha='center', va='top')

    # stimulation lines
    # ax_rhythm.vlines(x=t_stim, ymin=-0.1, ymax=1., color='gray', alpha=0.75, ls='--', linewidth=1.5, zorder=11, rasterized=False, clip_on=True)

    # stimulation lines
    ax_rhythm.axvline(x=t_stim, ymin=-1.5, ymax=1.1, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

    ax_common.axvline(x=t_stim, ymin=-0.5, ymax=1.5, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

    for ax in axs[2]:
        ax.axvline(x=t_stim, ymin=-0.5, ymax=1.5, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

    for ax in axs[3]:
        ln = ax.axvline(x=t_stim, ymin=-0.5, ymax=1.5, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

    ln.set_clip_on(True)

    # stimulation point
    ax_rhythm.scatter(x=t_stim, y=1.8, s=75, marker='v', edgecolors='white', facecolors='gray', rasterized=False, clip_on=False)


    # Shade the areas
    ax_rhythm.fill_betweenx(y=[0,1], x1=t_lims_adj[0], x2=t_lims_adj[1], color='k', alpha=0.25)
    axs[2][0].fill_betweenx(y=[0,N_scaling], x1=t_lims_adj[0], x2=t_lims_adj[1], color=c_exc, alpha=0.25)
    axs[2][0].fill_betweenx(y=[N_scaling+N_gap, ylims_rasters[1]], x1=t_lims_adj[0], x2=t_lims_adj[1], color=c_inh, alpha=0.25)

    # Set tight_layout
    G_outer.tight_layout(fig)
    # fig.tight_layout()


    """ Saving the figure """
    print('[+] Saving the figure...')
    fig.savefig(os.path.join(parent_dir, 'figures', 'supplementary', args.figure_name + '.png'), transparent=True, dpi=300, format='png')
    fig.savefig(os.path.join(parent_dir, 'figures', 'supplementary', args.figure_name + '.pdf'), transparent=True, dpi=300, format='pdf')

    print('[!] Done')

    # fig.tight_layout()
    plt.show()

    # Exit - no errors
    sys.exit(0)
