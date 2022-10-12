#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import json

import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.animation as animation
from scipy import signal as sig
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

fontprops = fm.FontProperties(size=12, family='monospace')

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42
mplb.rcParams['axes.titlesize'] = 11
mplb.rcParams['axes.labelsize'] = 8

def add_sizebar(ax, xlocs, ylocs, bcolor, text):
    """ Add a sizebar to the provided axis """
    ax.plot(xlocs, ylocs, ls='-', c=bcolor, linewidth=1., rasterized=False, clip_on=False)

    # add the text
    if type(text) is list:
        # bottom text
        ax.text(x=xlocs[0]+40*ms, y=ylocs[0], s=text[0], fontsize=fsize, va='center', ha='left', clip_on=False)

        # top text
        ax.text(x=xlocs[0]+40*ms, y=ylocs[1], s=text[1], fontsize=fsize, va='center', ha='left', clip_on=False)
    else:
        ax.text(x=xlocs[0]+40*ms, y=ylocs[0], s=text, fontsize=fsize, va='center', ha='left', clip_on=False)


def myround(x, base=100):
    """ Round x to closest *base* """
    val = int(base * round(x/base))
    return val if val > 0 else base*1


# main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate figure 2A from paper')

    parser.add_argument('-ra', '--rasters-all',
                        action='store_true',
                        default=False,
                        help='Set to plot all rasters instead of only for CA1.')

    parser.add_argument('-op', '--order-parameter',
                        action='store_true',
                        default=False,
                        help='Set to plot order parameter instead of phase.')

    parser.add_argument('--animate',
                        action='store_true',
                        default=False,
                        help='Set to generate an animation at the end.')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig2_A',
                        help='Name of the output figure [w/o file extension].')

    args = parser.parse_args()


    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Timing
    second = 1
    ms = 1e-3
    duration = 3.2*second
    dt = 0.1*ms
    fs = int(1/dt)
    winsize_FR = 5*ms
    overlap_FR = 0.9
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)
    binnum = int(duration/winsize_FR)
    t_stim = 2086.7*ms
    t_lims = [950*ms, 2700*ms] # ms
    # t_lims = [0*ms, 2000*ms] # ms
    interp = 'nearest'

    # Area names and sizes
    areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
    area_labels = ['EC', 'DG', 'CA3', 'CA1']
    N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'
    c_inh_RGB = np.array([191, 97, 106])/255
    c_exc_RGB = np.array([94, 129, 172])/255

    N = 256
    cvals_inh = np.ones((N,4))
    cvals_exc = np.ones((N,4))

    # inhibitory cmap
    reds = np.linspace(1., c_inh_RGB[0], N)
    greens = np.linspace(1., c_inh_RGB[1], N)
    blues = np.linspace(1., c_inh_RGB[2], N)
    cvals_inh[:,0] = reds[...]
    cvals_inh[:,1] = greens[...]
    cvals_inh[:,2] = blues[...]
    newcmap_inh = ListedColormap(cvals_inh)

    # excitatory cmap
    reds = np.linspace(1., c_exc_RGB[0], N)
    greens = np.linspace(1., c_exc_RGB[1], N)
    blues = np.linspace(1., c_exc_RGB[2], N)
    cvals_exc[:,0] = reds[...]
    cvals_exc[:,1] = greens[...]
    cvals_exc[:,2] = blues[...]
    newcmap_exc = ListedColormap(cvals_exc)

    # Raster downsampling
    N_scaling = 200
    N_gap = 10

    # Firing rates plotting gap
    rates_gap = 2 # Hz

    # Power spectra parameters
    window_size = 100*ms
    window_width = int(fs_FR*window_size) # samples
    overlap = 0.99 # percentage
    noverlap = int(overlap*window_width)

    # Set theta rhythm limits
    # xlims_rhythm = [t for t in t_lims]
    xlims_rhythm = [t for t in t_lims]
    ylims_rhythm = [-0.1, 1.4]

    # Set raster limits
    xlims_raster = [t for t in t_lims]
    ylims_raster = [0, 2*N_scaling+N_gap]

    # Set firing rate limits
    xlims_rates = [t for t in t_lims]
    ylims_rates = [-1, 500]

    # Set spectrogram limits
    xlims_freq = [t for t in t_lims]
    # xlims_freq[0] += window_size/2
    # xlims_freq[1] += window_size/2
    ylims_freq = [10, 150] # Hz
    zlims_freq = [0.1, 10] # cmap limits [vmin vmax]

    # Text parameters
    fsize = 9
    sizebar_off = 50*ms # sizebar offset

    """ Plot Figure 2 of the paper - TODO: Add DOI"""
    print('[+] Generating the figure...')

    # Figure sizes
    fig_width = 7.5
    fig_height = 5.4

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height))

    # Use gridspecs
    G_outer = GridSpec(5, 2, left=0.075, right=0.925, bottom=0.075, top=0.925,
                        wspace=0.05, hspace=0.5, height_ratios=(0.1, 0.1, 0.3, 0.2, 0.3), width_ratios=(0.99,0.01))
    G_rhythm = GridSpecFromSubplotSpec(1, 1, hspace=0.1, subplot_spec=G_outer[0,0])
    G_order_param = G_phase = GridSpecFromSubplotSpec(1, 1, hspace=0.1, subplot_spec=G_outer[1,0])
    if args.rasters_all:
        G_rasters = GridSpecFromSubplotSpec(4, 1, hspace=0.4, subplot_spec=G_outer[2,0])
    else:
        G_rasters = GridSpecFromSubplotSpec(1, 1, hspace=0.4, subplot_spec=G_outer[2,0])
    G_rates = GridSpecFromSubplotSpec(1, 1, hspace=0.6, subplot_spec=G_outer[3,0])
    G_specg = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_outer[4,0])
    G_specg_cbars = GridSpecFromSubplotSpec(1, 1, hspace=0.3, subplot_spec=G_outer[4,1])

    G_outer.tight_layout(fig)


    # Organize axes
    #------------------------
    axs = []

    # Rasters
    # ------------------------
    print('[>] Rasters')

    if args.rasters_all:
        ax0 = fig.add_subplot(G_rasters[0]) # EC E/I rasters
        ax1 = fig.add_subplot(G_rasters[1], sharex=ax0, sharey=ax0) # DG E/I rasters
        ax2 = fig.add_subplot(G_rasters[2], sharex=ax0, sharey=ax0) # CA3 E/I rasters
        ax3 = fig.add_subplot(G_rasters[3], sharex=ax0, sharey=ax0) # CA1 E/I rasters
        axs.append([ax0, ax1, ax2, ax3])

        # set label as area name
        ax0.set_title('Rasters')
        # ax0.set_title(area_labels[0])
        # ax1.set_title(area_labels[1])
        # ax2.set_title(area_labels[2])
        # ax3.set_title(area_labels[3])
        ax0.set_ylabel(areas[0][0].split('_')[0], rotation=0, labelpad=25.)
        ax1.set_ylabel(areas[1][0].split('_')[0], rotation=0, labelpad=25.)
        ax2.set_ylabel(areas[2][0].split('_')[0], rotation=0, labelpad=25.)
        ax3.set_ylabel(areas[3][0].split('_')[0], rotation=0, labelpad=25.)

        # set limits
        # ax0.set_xlim([575, (duration-450*ms)/winsize])
        ax0.set_xlim(xlims_raster)
        ax0.set_ylim(ylims_raster)

        # Hide x-y axes
        ax0.xaxis.set_visible(False)
        # ax0.yaxis.set_visible(False)
        ax1.xaxis.set_visible(False)
        # ax1.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
        # ax2.yaxis.set_visible(False)
        ax3.xaxis.set_visible(False)
        # ax3.yaxis.set_visible(False)

        # Hide some spines
        ax0.spines['top'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # Fix the ytick locators
        ax0.yaxis.set_major_locator(ticker.NullLocator())
        ax0.yaxis.set_minor_locator(ticker.NullLocator())
        ax1.yaxis.set_major_locator(ticker.NullLocator())
        ax1.yaxis.set_minor_locator(ticker.NullLocator())
        ax2.yaxis.set_major_locator(ticker.NullLocator())
        ax2.yaxis.set_minor_locator(ticker.NullLocator())
        ax3.yaxis.set_major_locator(ticker.NullLocator())
        ax3.yaxis.set_minor_locator(ticker.NullLocator())

        # Remove tick labels
        # ax0.xaxis.set_ticklabels([])
        ax0.yaxis.set_ticklabels([])
        # ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        # ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        # ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])

    else:
        ax0 = fig.add_subplot(G_rasters[0]) # CA1 E/I rasters
        axs.append([ax0])

        # set label as area name
        ax0.set_title('Rasters')
        ax0.set_ylabel(areas[3][0].split('_')[0], rotation=0, labelpad=-1.)

        # set limits
        ax0.set_xlim(xlims_raster)
        ax0.set_ylim(ylims_raster)

        # Hide x-y axes
        ax0.xaxis.set_visible(False)
        # ax0.yaxis.set_visible(False)

        # Hide some spines
        ax0.spines['top'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        ax0.spines['right'].set_visible(False)

        # Fix the ytick locators
        ax0.yaxis.set_major_locator(ticker.NullLocator())
        ax0.yaxis.set_minor_locator(ticker.NullLocator())

        # Remove tick labels
        # ax0.xaxis.set_ticklabels([])
        ax0.yaxis.set_ticklabels([])


    # Firing Rates
    # ------------------------
    # ax_rate_inh = fig.add_subplot(G_rates[0], sharex=ax0)
    # ax_rate_exc = fig.add_subplot(G_rates[1], sharex=ax0, sharey=ax_rate_inh)
    # axs.append([ax_rate_exc, ax_rate_inh])
    ax_rates = fig.add_subplot(G_rates[0], sharex=ax0)
    axs.append([ax_rates])

    # Set the title
    ax_rates.set_title('CA1 Firing Rates')

    # Set the x-y limits
    ax_rates.set_ylim(ylims_rates)
    # ax_rates.set_ylim([0, 400])

    # Set the ticks
    ax_rate_majors = np.arange(0., t_lims[1]/second, .5) #[0.5, 1.0, 1.5...]
    # ax_rate_exc.xaxis.set_major_locator(ticker.FixedLocator(ax_rate_majors))
    # ax_rate_exc.xaxis.set_minor_locator(ticker.NullLocator())
    ax_rates.xaxis.set_major_locator(ticker.NullLocator())
    ax_rates.xaxis.set_minor_locator(ticker.NullLocator())

    # Hide x-y axes
    # # ax_rate_exc.xaxis.set_visible(False)
    # ax_rate_exc.yaxis.set_visible(False)
    # ax_rate_inh.xaxis.set_visible(False)
    # ax_rate_inh.yaxis.set_visible(False)
    ax_rates.yaxis.set_visible(False)

    # Hide some spines
    # ax_rate_exc.spines['top'].set_visible(False)
    # # ax_rate_exc.spines['bottom'].set_visible(False)
    # ax_rate_exc.spines['left'].set_visible(False)
    # ax_rate_exc.spines['right'].set_visible(False)
    # ax_rate_inh.spines['top'].set_visible(False)
    # ax_rate_inh.spines['bottom'].set_visible(False)
    # ax_rate_inh.spines['left'].set_visible(False)
    # ax_rate_inh.spines['right'].set_visible(False)
    ax_rates.spines['top'].set_visible(False)
    ax_rates.spines['bottom'].set_visible(False)
    ax_rates.spines['left'].set_visible(False)
    ax_rates.spines['right'].set_visible(False)


    # Spectrograms
    # ------------------------
    print('[>] Spectrograms')
    ax_specg_inh = fig.add_subplot(G_specg[0])
    ax_specg_exc = fig.add_subplot(G_specg[1])
    axs.append([ax_specg_exc, ax_specg_inh])

    # Set the title
    ax_specg_inh.set_title('Spectrograms')

    # # Set the x-y limits
    # ax_specg_inh.set_xlim(xlims_freq)
    # ax_specg_inh.set_ylim(ylims_freq)
    # ax_specg_exc.set_xlim(xlims_freq)
    # ax_specg_exc.set_ylim(ylims_freq)

    # Set the ticks
    specg_freq_majors = [10, 60, 120]
    ax_specg_inh.yaxis.set_major_locator(ticker.FixedLocator(specg_freq_majors))
    ax_specg_exc.yaxis.set_major_locator(ticker.FixedLocator(specg_freq_majors))

    specg_freq_majors = np.arange(0., t_lims[1]/second, .5) #[0.5, 0.6, 0.7, 1., 1.25, 1.5]
    specg_freq_minors = np.arange(0., t_lims[1]/second, .1)
    ax_specg_exc.xaxis.set_major_locator(ticker.FixedLocator(specg_freq_majors))
    ax_specg_exc.xaxis.set_minor_locator(ticker.FixedLocator(specg_freq_minors))

    # tick sizes
    ax_specg_exc.tick_params(axis='x', which='major', width=1.0)
    ax_specg_exc.tick_params(axis='x', which='major', length=10)
    ax_specg_exc.tick_params(axis='x', which='minor', width=1.0)
    ax_specg_exc.tick_params(axis='x', which='minor', length=5, labelcolor='0.25')

    ax_specg_exc.tick_params(axis='both', which='both', labelsize=9)
    ax_specg_inh.tick_params(axis='both', which='both', labelsize=9)

    # Hide x axis for inh
    ax_specg_inh.xaxis.set_visible(False)

    # Set xlabel
    ax_specg_exc.set_xlabel('Time [s]', labelpad=-5.)

    # Hide some spines
    ax_specg_exc.spines['top'].set_visible(False)
    # ax_specg_exc.spines['bottom'].set_visible(False)
    # ax_specg_exc.spines['left'].set_visible(False)
    ax_specg_exc.spines['right'].set_visible(False)
    ax_specg_inh.spines['top'].set_visible(False)
    # ax_specg_inh.spines['bottom'].set_visible(False)
    # ax_specg_inh.spines['left'].set_visible(False)
    ax_specg_inh.spines['right'].set_visible(False)


    # Order Parameter / Phase
    # ------------------------
    if args.order_parameter:
        print('[>] Order Parameter')
        ax_common = fig.add_subplot(G_order_param[0])
        xlims_common = xlims_rhythm
        ylims_common = ylims_rhythm

        # Set the title
        ax_common.set_title('Order Parameter')

    else:
        print('[>] Phase')
        ax_common = fig.add_subplot(G_phase[0])
        xlims_common = xlims_rhythm
        # ylims_common = [-np.pi-0.1, np.pi+0.1]
        ylims_common = [-0.1, 2*np.pi+0.1]

        # Set the title
        ax_common.set_title('Phase')

    axs.append(ax_common)

    # Set the x-y limits
    ax_common.set_xlim(xlims_common)
    ax_common.set_ylim(ylims_common)

    # Set x-y ticks
    # ax_common_majors = np.arange(0., (duration+50*ms)/second, .5)
    # ax_common.xaxis.set_major_locator(ticker.FixedLocator(ax_common_majors))
    # ax_common.xaxis.set_minor_locator(ticker.NullLocator())

    # Hide x-y axes
    ax_common.xaxis.set_visible(False)
    ax_common.yaxis.set_visible(False)

    # Hide some spines
    ax_common.spines['top'].set_visible(False)
    ax_common.spines['bottom'].set_visible(False)
    ax_common.spines['left'].set_visible(False)
    ax_common.spines['right'].set_visible(False)


    # Rhythm
    # ------------------------
    print('[>] Input rhythm')
    ax_rhythm = fig.add_subplot(G_rhythm[0])
    axs.append(ax_rhythm)

    # Set the title
    ax_rhythm.set_title('Theta Rhythm')

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


    # Other stuff
    # ------------------------
    # legend patches
    # print('[>] Legend')
    # leg_lines = [Line2D([0], [0], color=c_inh, lw=4),
                # Line2D([0], [0], color=c_exc, lw=4)]
    # labels = ['Inhibitory', 'Excitatory']
    # fig.legend(leg_lines, labels, bbox_transform=fig.transFigure, bbox_to_anchor=(0.15,0.5), loc='upper right', ncol=1)
    # fig.legend(leg_lines, labels, loc='upper right', ncol=1)


    # Iterate over areas
    print('[+] Actually plotting stuff...')

    for area_idx in range(len(areas)):
        if not args.rasters_all and area_idx<3:
            continue;

        # load t-i arrays for this area
        print('[+] Loading the spikes for area', areas[area_idx][0].split('_')[0])
        i_exc = np.loadtxt(os.path.join(parent_dir, 'results', 'analysis', 'current', 'desc2', 'data', 'spikes/{0}_spikemon_i.txt'.format(areas[area_idx][0])))
        t_exc = np.loadtxt(os.path.join(parent_dir, 'results', 'analysis', 'current', 'desc2', 'data', 'spikes/{0}_spikemon_t.txt'.format(areas[area_idx][0])))
        i_inh = np.loadtxt(os.path.join(parent_dir, 'results', 'analysis', 'current', 'desc2', 'data', 'spikes/{0}_spikemon_i.txt'.format(areas[area_idx][1])))
        t_inh = np.loadtxt(os.path.join(parent_dir, 'results', 'analysis', 'current', 'desc2', 'data', 'spikes/{0}_spikemon_t.txt'.format(areas[area_idx][1])))

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
        N_exc = N_tot[area_idx][0]
        N_inh = N_tot[area_idx][1]

        # select some neurons randomly, subscaling
        # exc_mixed = np.random.permutation(np.arange(N_exc))[:N_scaling]
        # inh_mixed = np.random.permutation(np.arange(N_inh))[:N_scaling]
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

        # select axis
        if args.rasters_all:
            ax_curr = axs[0][area_idx]
        else:
            ax_curr = axs[0][0]

        # inhibitory
        # ax_curr.plot(t_inh_sub, i_inh_sub, 'o', c=c_inh, markersize=.25, alpha=.75, zorder=1, rasterized=True)
        ax_curr.scatter(t_inh_sub, i_inh_sub, s=0.55, linewidth=1., marker='|', c=c_inh, edgecolors=None, alpha=1., rasterized=True)
        # ax_curr.set_rasterization_zorder(2)
        # ax_curr.set_rasterized(True)

        # excitatory
        # ax_curr.plot(t_exc_sub, i_exc_sub, 'o', c=c_exc, markersize=.25, alpha=.75, zorder=1, rasterized=True)
        ax_curr.scatter(t_exc_sub, i_exc_sub, s=0.55, linewidth=1., marker='|', c=c_exc, edgecolors=None, alpha=1., rasterized=True)
        # ax_curr.set_rasterization_zorder(2)
        # ax_curr.set_rasterized(True)

        # Calculate mean firing rates
        t_lims_adj = [1000*ms, 2000*ms]
        duration_adj = t_lims_adj[1] - t_lims_adj[0]

        # FR_inh_mean = (len(t_inh)/duration)/N_inh
        # FR_exc_mean = (len(t_exc)/duration)/N_exc
        FR_inh_mean = (sum((t_inh>=t_lims_adj[0]) & (t_inh<t_lims_adj[1]))/duration_adj)/N_inh
        FR_exc_mean = (sum((t_exc>=t_lims_adj[0]) & (t_exc<t_lims_adj[1]))/duration_adj)/N_exc

        # add it as a text
        ax_curr.text(x=xlims_rates[1]+100*ms, y=1.5*N_scaling+N_gap, s=r'$\mu_I$: {0:.1f} Hz'.format(FR_inh_mean), fontsize=fsize, ha='center', color=c_inh, clip_on=False)
        ax_curr.text(x=xlims_rates[1]+100*ms, y=N_scaling//2, s=r'$\mu_E$: {0:.1f} Hz'.format(FR_exc_mean), fontsize=fsize, ha='center', color=c_exc, clip_on=False)

        # Shade the areas
        ax_curr.fill_betweenx(y=[0,N_scaling], x1=t_lims_adj[0], x2=t_lims_adj[1], cmap=newcmap_exc, alpha=0.1)
        ax_curr.fill_betweenx(y=[N_scaling+N_gap, ylims_raster[1]], x1=t_lims_adj[0], x2=t_lims_adj[1], cmap=newcmap_inh, alpha=0.1)

    # # add a sizebar for the y-axis
    # add_sizebar(axs[0][3], [t_lims[1]+20*ms, t_lims[1]+20*ms], [0, 100], 'black', '100pts')


    # ==================
    # Plot the input rhythm
    # ==================
    print('[+] Plotting rhythm...')

    rhythm = np.loadtxt(os.path.join(parent_dir, 'results', 'analysis', 'current', 'desc2','data', 'order_param_mon_rhythm.txt'))
    ax_rhythm.plot(np.arange(0.,duration,dt), rhythm/(np.max(rhythm)), ls='-', c='k', linewidth=1.2, rasterized=False, zorder=1)

    # vertical lines at x-points
    pks, _ = sig.find_peaks(rhythm, distance=int(80*ms*fs))
    fval = 1/(np.mean(pks[1:] - pks[0:-1])/fs) if len(pks)>1 else 1/(pks[0]/fs)

    # for peak in pks:
        # if (peak*dt >= t_lims[0]) & (peak*dt <= t_lims[1]):
            # "X" marks the spot
            # ax_rhythm.plot(peak*dt, rhythm[peak], 'C1x', markersize=12, zorder=10, clip_on=False)

            # trace the treasure
            # ax_rhythm.vlines(x=peak*dt, ymin=-15.85, ymax=rhythm[peak], color='black', ls='--', linewidth=0.5, zorder=11, clip_on=False)

    # stimulation line
    ax_rhythm.vlines(x=t_stim, ymin=-19.6, ymax=1., color='gray', alpha=0.75, ls='--', linewidth=1.5, zorder=11, rasterized=False, clip_on=False)

    # stimulation point
    ax_rhythm.scatter(x=t_stim, y=1.3, s=55, marker='v', edgecolors='white', facecolors='gray', rasterized=False, clip_on=False)

    # ax_rhythm.annotate('Stimulation Pulse', xy=(t_stim, 1.2), xytext=(t_stim, 2.5), arrowprops=dict(facecolor='red', shrink=0.05))

    # text frequency label
    ax_rhythm.text(x=xlims_rhythm[0]+10*ms, y=1.2, s=r"$f_\theta={0:.1f}$ Hz".format(fval), fontsize=fsize, ha='left', color='k', clip_on=False)

    # add a sizebar for the y-axis
    add_sizebar(ax_rhythm, [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 1.], 'black', ['0', '1'])


    # ==================
    # Plot the phase / order parameter
    # ==================
    if args.order_parameter:
        print('[+] Plotting order parameter...')
        data = np.loadtxt(os.path.join(parent_dir, 'results', 'analysis', 'current', 'desc2', 'data','order_param_mon_coherence.txt'))

        # asymptote
        ax_common.hlines(y=1., xmin=0., xmax=duration, color='k', ls='--', linewidth=0.5, zorder=11)

        # add a sizebar for the y-axis
        add_sizebar(ax_common, [xlims_common[1]+sizebar_off, xlims_common[1]+sizebar_off], [0, 1.], 'black', '1.pt')

    else:
        print('[+] Plotting phase...')
        data = np.loadtxt(os.path.join(parent_dir, 'results', 'analysis', 'current', 'desc2', 'data','order_param_mon_phase.txt'))
        # data = (data + np.pi) % (2 * np.pi)
        data += (1.*(data<0)*2*np.pi)

        # add a sizebar for the y-axis
        add_sizebar(ax_common, [xlims_common[1]+sizebar_off, xlims_common[1]+sizebar_off], [0, 2*np.pi], 'black', ['0', '$2\pi$'])

        # text with stimulation phase [deg/rad]
        ax_common.scatter(x=t_stim, y=data[int(t_stim*fs)], s=12, marker='o', c='k')
        ax_common.text(x=t_stim-75*ms, y=data[int(t_stim*fs)]+0.25, s=r"$\pi/2$", fontsize=fsize, ha='left', color='k', clip_on=False)

    # Data plotting
    ax_common.plot(np.arange(0.,duration,dt), data, ls='-', c='k', linewidth=1.2, rasterized=False, zorder=1)


    # ==================
    # Plot CA1 FRs
    # ==================
    tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    # SAVE THE NON-NORMALIZED FR SIGNALS
    np.savetxt(os.path.join(parent_dir, 'test_FRs', 'FR_inh.txt'), FR_inh, fmt='%.8f')
    np.savetxt(os.path.join(parent_dir, 'test_FRs', 'tv_inh.txt'), tv_inh_FR, fmt='%.8f')
    np.savetxt(os.path.join(parent_dir, 'test_FRs', 'FR_exc.txt'), FR_exc, fmt='%.8f')
    np.savetxt(os.path.join(parent_dir, 'test_FRs', 'tv_exc.txt'), tv_exc_FR, fmt='%.8f')

    # Normalize the FRs
    FR_inh_norm = (FR_inh/winsize_FR)/N_inh
    FR_exc_norm = (FR_exc/winsize_FR)/N_exc

    # Plot the FRs
    # ax_rate_inh.plot(tv_inh_FR, FR_inh_norm, ls='-', linewidth=1., c=c_inh, label='inh', zorder=10, rasterized=False)
    # ax_rate_exc.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1., c=c_exc, label='exc', zorder=10, rasterized=False)
    ax_rates.plot(tv_inh_FR, FR_inh_norm+FR_exc_norm.max()+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
    ax_rates.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)

    # add labels
    # ax_rate_inh.set_title('Inhibitory', color=c_inh, loc='left')
    # ax_rate_exc.set_title('Excitatory', color=c_exc, loc='left')

    # ax_rate_inh.text(x=-10*ms, y=ylims_rates[1]//2, s='Inhibitory', ha='center', color=c_inh, clip_on=False)
    # ax_rate_exc.text(x=-10*ms, y=ylims_rates[1]//2, s='Excitatory', ha='center', color=c_exc, clip_on=False)
    # ax_rates.text(x=xlims_rates[0]-10*ms, y=ylims_rates[1]-100, s='Inhibitory', fontsize=fsize, ha='center', color=c_inh, clip_on=False)
    ax_rates.text(x=xlims_rates[0]-50*ms, y=ylims_rates[0]+150+FR_exc_norm.max()+rates_gap, s='Inhibitory', fontsize=fsize, ha='center', color=c_inh, clip_on=False)
    ax_rates.text(x=xlims_rates[0]-50*ms, y=ylims_rates[0]+75, s='Excitatory', fontsize=fsize, ha='center', color=c_exc, clip_on=False)

    # add a sizebar for the y-axis
    # add_sizebar(ax_rate_exc, [duration+sizebar_off, duration+sizebar_off], [0, 50], 'black', '50Hz')
    add_sizebar(ax_rates, [xlims_rates[1]+sizebar_off, xlims_rates[1]+sizebar_off], [0, 100], 'black', '100Hz')


    # ==================
    # Plot the spectrograms
    # ==================
    specgram_kwargs = { 'return_onesided' : True,
                        'scaling' : 'density',
                        'mode' : 'magnitude' }
    fv_inh, tv_inh, pspec_inh = my_specgram(signal=FR_inh_norm,
                                fs=fs_FR,
                                window_width=window_width,
                                window_overlap=noverlap,
                                **specgram_kwargs)

    fv_exc, tv_exc, pspec_exc = my_specgram(signal=FR_exc_norm,
                                fs=fs_FR,
                                window_width=window_width,
                                window_overlap=noverlap,
                                **specgram_kwargs)

    # avoid division by zero in log transform
    pspec_inh[np.where(pspec_inh<1e-10)] = 1e-10
    pspec_exc[np.where(pspec_exc<1e-10)] = 1e-10

    # get log power
    # pspec_inh_dB = 10*np.log10(pspec_inh)
    # pspec_exc_dB = 10*np.log10(pspec_exc)

    # find min/max for plotting
    # maxval_dB = max(pspec_inh_dB.max(), pspec_exc_dB.max())
    # minval_dB = min(pspec_inh_dB.min(), pspec_exc_dB.min()).
    # norm = colors.Normalize(vmin=minval_dB, vmax=maxval_dB)

    # pcm_inh = ax_specg_inh.pcolormesh(tv_inh, fv_inh, pspec_inh_dB, vmin=minval_dB, vmax=maxval_dB, cmap=newcmap_inh, shading='gouraud')
    # pcm_exc = ax_specg_exc.pcolormesh(tv_exc, fv_exc, pspec_exc_dB, vmin=minval_dB, vmax=maxval_dB, cmap=newcmap_exc, shading='gouraud')

    # set vmin/vmax for plotting
    vmin = 1e-12
    vmax = 1.

    # normalization of colors in [vmin, vmax]
    # norm_inh = colors.Normalize(vmin=-1, vmax=1)
    # norm_exc = colors.Normalize(vmin=-1, vmax=1)
    # norm_inh = colors.LogNorm(vmin=pspec_inh.min(), vmax=pspec_inh.max())
    # norm_exc = colors.LogNorm(vmin=pspec_exc.min(), vmax=pspec_exc.max())
    norm_inh = colors.Normalize(vmin=vmin, vmax=vmax)
    norm_exc = colors.Normalize(vmin=vmin, vmax=vmax)
    # norm_com = colors.Normalize(vmin=1e-12, vmax=.5)

    # Plot as images
    fidx = np.logical_and(fv_inh>=40, fv_inh<=120)

    # plt.imshow(pspec_exc[fidx,:], aspect='auto', origin='lower', extent=[tv_exc.min(), tv_exc.max(), fv_exc[fidx].min(), fv_exc[fidx].max()], interpolation='nearest')

    # joint normalization - min(exc,inh) / max(exc,inh)
    vlow = min(pspec_inh.min(), pspec_exc.min())
    vhigh = max(pspec_inh.max(), pspec_exc.max())
    im_inh = ax_specg_inh.pcolormesh(tv_inh, fv_inh, pspec_inh/vhigh, cmap='inferno', norm=norm_inh, shading='auto', rasterized=True)
    im_exc = ax_specg_exc.pcolormesh(tv_exc, fv_exc, pspec_exc/vhigh, cmap='inferno', norm=norm_exc, shading='auto', rasterized=True)

    # ax_specg_inh.set_ylabel('Inhibitory', color=c_inh)
    # ax_specg_exc.set_ylabel('Excitatory', color=c_exc)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=1.)

    # place a text box in upper left in axes coords
    # ax_specg_inh.text(0.05, 0.95, 'inhibitory', transform=ax_specg_inh.transAxes, verticalalignment='top', bbox=props)
    # ax_specg_inh.text(0.01, 0.9, 'I', fontsize=14, transform=ax_specg_inh.transAxes, color='white', verticalalignment='top')
    # ax_specg_exc.text(0.01, 0.9, 'E', fontsize=14, transform=ax_specg_exc.transAxes, color='white', verticalalignment='top')

    # w/ white box
    # ax_specg_inh.text(0.02, 0.9, 'Inhibitory', fontsize=11, transform=ax_specg_inh.transAxes, color=c_inh, verticalalignment='top', bbox=props)
    # ax_specg_exc.text(0.02, 0.9, 'Excitatory', fontsize=11, transform=ax_specg_exc.transAxes, color=c_exc, verticalalignment='top', bbox=props)

    # w/ white outline
    # ax_specg_inh.text(0.02, 0.9, 'Inhibitory', fontsize=fsize, transform=ax_specg_inh.transAxes, color=c_inh, verticalalignment='top', path_effects=[patheffects.withStroke(linewidth=1, foreground='white', capstyle="round")])
    # ax_specg_exc.text(0.02, 0.9, 'Excitatory', fontsize=fsize, transform=ax_specg_exc.transAxes, color=c_exc, verticalalignment='top', path_effects=[patheffects.withStroke(linewidth=1, foreground='white', capstyle="round")])

    # w/ white text
    ax_specg_inh.text(0.02, 0.9, 'Inhibitory', fontsize=fsize, transform=ax_specg_inh.transAxes, color='white', verticalalignment='top')
    ax_specg_exc.text(0.02, 0.9, 'Excitatory', fontsize=fsize, transform=ax_specg_exc.transAxes, color='white', verticalalignment='top')


    # data_inh, freqs_inh, bins_inh, im_inh = ax_specg_inh.specgram(FR_inh_norm, NFFT=window_width, pad_to=2048, Fs=fs_FR, noverlap=noverlap, scale='linear', window=sig.windows.hann(M=window_width, sym=False), cmap=newcmap_inh)
    # # ax_specg_inh.axis('tight')
    #
    # data_exc, freqs_exc, bins_exc, im_exc = ax_specg_exc.specgram(FR_exc_norm, NFFT=window_width, pad_to=2048, Fs=fs_FR, noverlap=noverlap, scale='linear', window=sig.windows.hann(M=window_width, sym=False), cmap=newcmap_exc)
    # ax_specg_exc.axis('tight')

    # vmin=0., vmax=0.2
    # im_inh = ax_specg_inh.imshow(pspec_inh/pspec_inh.max(), vmin=vmin, vmax=vmax, origin='lower', aspect='auto', extent=[tv_inh[0], tv_inh[-1], fv_inh[0], fv_inh[-1]], cmap=newcmap_inh, interpolation='nearest')
    # im_exc = ax_specg_exc.imshow(pspec_exc/pspec_inh.max(), vmin=vmin, vmax=vmax, origin='lower', aspect='auto', extent=[tv_exc[0], tv_exc[-1], fv_exc[0], fv_exc[-1]], cmap=newcmap_exc, interpolation='nearest')

    # Set the x-y limits
    ax_specg_inh.set_xlim(xlims_freq)
    ax_specg_inh.set_ylim(ylims_freq)
    ax_specg_exc.set_xlim(xlims_freq)
    ax_specg_exc.set_ylim(ylims_freq)

    # pcm_inh= ax_specg_inh.specgram(FR_exc_norm, NFFT=window_width, detrend='none', Fs=fs_FR, window=sig.windows.hann(M=window_width, sym=False), noverlap=noverlap, scale_by_freq=False, mode='magnitude', scale='dB', sides='onesided')

    # Make sure the spectrogams are rasterized!
    # ax_specg_inh.set_rasterized(True)
    # ax_specg_exc.set_rasterized(True)

    # Colorbars
    # fig.colorbar(pcm_exc, ax=ax_specg_exc)
    # fig.colorbar(pcm_inh, ax=ax_specg_inh)
    # cbar_inh_ax = fig.add_subplot(G_specg_cbars[0])
    # cbar_exc_ax = fig.add_subplot(G_specg_cbars[1])

    # cbi = fig.colorbar(im_inh, cax=cbar_inh_ax, aspect=1, ticks=[.5])
    # cbe = fig.colorbar(im_exc, cax=cbar_exc_ax, aspect=1, ticks=[.5])
    #
    # cbi.outline.set_color('black')
    # cbi.outline.set_linewidth(0.5)
    # cbi.solids.set_rasterized(True)
    # cbe.outline.set_color('black')
    # cbe.outline.set_linewidth(0.5)
    # cbe.solids.set_rasterized(True)
    #
    # cbe.dividers.set_color('none')
    # cbe.dividers.set_linewidth(5)

    # Add the colorbar
    cbar_comm = fig.add_subplot(G_specg_cbars[0])
    cbe = fig.colorbar(im_exc, cax=cbar_comm, aspect=1, ticks=[0., vmax])
    cbe.outline.set_color('black')
    cbe.outline.set_linewidth(0.5)
    cbe.solids.set_rasterized(True)
    cbe.dividers.set_color('none')
    cbe.dividers.set_linewidth(5)
    cbe.ax.tick_params(labelsize=9)

    # sizebars
    # ax_specg_exc.plot([550*ms, 600*ms, None, 550*ms, 550*ms], [-60, -60, None, 0, 0], ls='-', c='r', linewidth=1., rasterized=True, clip_on=False)
    # ax_specg_exc.text(x=575*ms, y=-100, s='50ms', ha='center', fontproperties=fontprops, clip_on=False)

    # save the figure
    print('[+] Saving the figures...')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig2', args.figure_name + '.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig2', args.figure_name + '.pdf'), transparent=True, dpi=300, format='pdf', pad_inches=0)


    # Also make an animation
    # ------------------------
    if args.animate:
        x_min = 0
        x_max = int(duration/dt)
        x_vals = range(x_min, x_max+1) # possible x values for the line

        animation_time = 5*second
        framerate = 60
        F = animation_time*framerate
        t_step = int(x_max/F)

        moving_line = ax_rhythm.axvline(x=[0], ymin=-11., ymax=1., color='red', ls='-', linewidth=1.5, zorder=12, clip_on=False)

        def update_line(i):
            t_val = x_vals[i*t_step]
            moving_line.set_xdata([t_val*dt, t_val*dt])
            # moving_line.set_ydata([-15.85, 1.])
            return moving_line,

        # create animation using the animate() function
        print('[+] Making animation...')
        line_animation = animation.FuncAnimation(fig, update_line, frames=F, interval=1e3/framerate, blit=True)

        print('[+] Saving the video...')
        line_animation.save(os.path.join(parent_dir, 'figures', 'test.mp4'), fps=framerate, extra_args=['-vcodec', 'libx264'])
        # save animation at 30 frames per second
        # line_animation.save('/home/nikos/Documents/projects/Python/memstim-hh/figures/test.gif', writer='imagemagick', fps=framerate)

    print('[!] Done')

    # fig.tight_layout()
    plt.show()

    # Exit - no errors
    sys.exit(0)
