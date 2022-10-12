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

# TensorPAC stuff
from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42


def add_sizebar(ax, xlocs, ylocs, bcolor, text, textx, texty, rot, ha, va):
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



def check_MI(signal, theta_freqs=[4, 12], gamma_freqs=[30, 80], return_distance=False):
    """
        PAC check using Modulation Index (MI)
          Wrapper function that filters and analyzes the input signal for phase-amplitude coupling (PAC) using the Modulation Index (MI) - ported function from Matlab.
    """
    # 1. Filtering at theta/gamma bands
    sig_filt_gamma = butter_bandpass_filter(signal, gamma_freqs[0], gamma_freqs[1], fs_FR, sos=True)
    sig_filt_theta = butter_bandpass_filter(signal, theta_freqs[0], theta_freqs[1], fs_FR, sos=True)

    # 2. Time series of phases / amplitudes using Hilbert Transform
    xfp = sig.hilbert(sig_filt_theta)
    xfA = sig.hilbert(sig_filt_gamma)
    sig_amp = np.abs(xfA)
    sig_phase = np.angle(xfp)

    # 3./4. Calculate Modulation Index (MI)
    MI, dist_KL = my_modulation_index(sig_phase, sig_amp)

    if return_distance:
        return MI, dist_KL
    else:
        return MI


# main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate figure 4A from paper')

    parser.add_argument('-ra', '--rasters-all',
                        action='store_true',
                        default=False,
                        help='Set to plot all rasters instead of only for CA1.')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig4_A',
                        help='Name of the output figure [w/o file extension]')

    args = parser.parse_args()

    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Timing
    second = 1
    ms = 1e-3
    duration = 3*second
    dt = 0.1*ms
    fs = int(1*second/dt)
    tv = np.linspace(0, duration, fs*duration)/ms
    winsize_FR = 5*ms
    overlap_FR = 0.9
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)

    # Stim onset
    t_stim = 1.8*second/ms

    # Pre/Post stim PAC
    tlims_pre = np.array([-1000, 0]) + t_stim
    tlims_post = np.array([0, 1000]) + t_stim

    # Directories
    fig4_dir = os.path.join(parent_dir, 'res_test_fig4', '10.0_nA', '0.00_1800.0_ms', '11-10-2022 15H02M22S')
    fig4_data = os.path.join(fig4_dir, 'data')
    fig4_currents = os.path.join(fig4_data, 'currents')
    fig4_spikes = os.path.join(fig4_data, 'spikes')

    # Area names and sizes
    areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
    area_labels = ['EC', 'DG', 'CA3', 'CA1']
    N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'

    # Figure sizes
    fig_width = 5.5
    fig_height = 4.0
    if args.rasters_all:
        fig_height = 8.5

    # Rasters
    N_scaling = 200
    N_gap = 5

    # Firing rates plotting gap
    rates_gap = 15 # Hz

    # Text parameters
    fsize_ticks = fsize_legends = 8
    fsize_xylabels = 9
    fsize_figtitles = 11
    sizebar_off = 80 # sizebar offset

    # ax parameters
    xlims_rhythm = np.array([-500, 1000]) + t_stim
    ylims_rhythm = [0,1]

    xlims_raster = xlims_rhythm
    ylims_raster = [0, 2*N_scaling+N_gap+1]


    """ Plot Figure 4 of the paper - TODO: Add DOI """
    print('[+] Generating the figure...')

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height), tight_layout=True)

    # Using GridSpecs
    if args.rasters_all:
        G_outer = GridSpec(4, 1, #left=0.1, right=0.95, bottom=0.15, top=0.925,
                           height_ratios=(0.1, 0.4, 0.1, 0.4), figure=fig)
        G_rasters = G_outer[1].subgridspec(4, 1)
    else:
        G_outer = GridSpec(4, 1, #left=0.1, right=0.95, bottom=0.15, top=0.925,
                           height_ratios=(0.1, 0.2, 0.2, 0.5), figure=fig)
        G_rasters = G_outer[1]

    G_rhythm = G_outer[0]
    G_FRs = G_outer[2]
    G_PAC = G_outer[3].subgridspec(1, 2)

    G_outer.tight_layout(fig)


    # Organize axes
    #------------------------
    axs = []

    # Axis for theta rhythm
    ax_rhythm = fig.add_subplot(G_rhythm)
    axs.insert(0, [ax_rhythm])

    # set label as area name
    # ax_rhythm.set_title(r'Input $\theta$ rhythm', loc='center', fontsize=fsize_figtitles)

    # Set the title
    ax_rhythm.set_title('Theta Rhythm', loc='center', fontsize=fsize_figtitles)

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

    # Axes for rasters
    if args.rasters_all:
        ax0 = fig.add_subplot(G_rasters[0], sharex=ax_rhythm) # EC E/I rasters
        ax1 = fig.add_subplot(G_rasters[1], sharex=ax_rhythm, sharey=ax0) # DG E/I rasters
        ax2 = fig.add_subplot(G_rasters[2], sharex=ax_rhythm, sharey=ax0) # CA3 E/I rasters
        ax3 = fig.add_subplot(G_rasters[3], sharex=ax_rhythm, sharey=ax0) # CA1 E/I rasters
        axs.insert(1, [ax0, ax1, ax2, ax3])

        # Set area labels
        ax0.set_ylabel(areas[0][0].split('_')[0], rotation=0, labelpad=20., fontsize=fsize_xylabels)
        ax1.set_ylabel(areas[1][0].split('_')[0], rotation=0, labelpad=20., fontsize=fsize_xylabels)
        ax2.set_ylabel(areas[2][0].split('_')[0], rotation=0, labelpad=20., fontsize=fsize_xylabels)
        ax3.set_ylabel(areas[3][0].split('_')[0], rotation=0, labelpad=20., fontsize=fsize_xylabels)

        # Set x-lims
        ax0.set_xlim(xlims_raster)

        # Set y-lims
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
        ax0 = fig.add_subplot(G_rasters, sharex=ax_rhythm)
        axs.insert(1, [ax0])

        ax0.set_ylabel(areas[3][0].split('_')[0], rotation=0, labelpad=20., fontsize=fsize_xylabels)

        # Set x-lims
        ax0.set_xlim(xlims_raster)

        # Set y-lims
        ax0.set_ylim(ylims_raster)

        # Hide x-y axes
        ax0.xaxis.set_visible(False)
        ax0.yaxis.set_visible(False)

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


    # Axis for CA1 FRs
    ax_FRs = fig.add_subplot(G_FRs, sharex=ax_rhythm)
    axs.insert(2, [ax_FRs])

    # Hide x-y axes
    ax_FRs.xaxis.set_visible(False)
    ax_FRs.yaxis.set_visible(False)

    # Hide some spines
    ax_FRs.spines['top'].set_visible(False)
    ax_FRs.spines['bottom'].set_visible(False)
    ax_FRs.spines['left'].set_visible(False)
    ax_FRs.spines['right'].set_visible(False)

    # Fix the ytick locators
    ax_FRs.yaxis.set_major_locator(ticker.NullLocator())
    ax_FRs.yaxis.set_minor_locator(ticker.NullLocator())

    # Remove tick labels
    ax_FRs.xaxis.set_ticklabels([])
    ax_FRs.yaxis.set_ticklabels([])

    # Axes for pre/post PAC
    ax_PAC_pre = fig.add_subplot(G_PAC[0])
    ax_PAC_post = fig.add_subplot(G_PAC[1])
    axs.insert(3, [ax_PAC_pre, ax_PAC_post])


    # Load and plot the data (panel A)
    #------------------------
    print('[+] Loading theta rhythm...')
    rhythm = np.loadtxt(os.path.join(fig4_data, 'order_param_mon_rhythm.txt'))

    print('[>] Plotting panel A - theta rhythm')
    ax_rhythm.plot(tv, rhythm/(np.max(rhythm)), ls='-', c='k', linewidth=1.2, rasterized=False, zorder=1)

    # Don't forget to mark the stimulation onset!
    # add marker for stimulation onset
    ax_rhythm.scatter(x=t_stim, y=1.25, s=12.5, linewidth=1., marker='v', c='gray', edgecolors=None, alpha=1, rasterized=False, clip_on=False)

    # stimulation line
    ax_rhythm.axvline(x=t_stim, ymin=-0.5, ymax=1.1, color='gray', alpha=0.75, ls='--', linewidth=1.5, zorder=10, rasterized=False, clip_on=False)

    # Spikes
    print('[+] Loading rasters...')

    for area_idx in range(len(areas)):
        if not args.rasters_all and area_idx<3:
            continue;

        # load t-i arrays for this area
        print('[+] Loading spikes for area', areas[area_idx][0].split('_')[0])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)

            # 10nA
            i_exc = np.loadtxt(os.path.join(fig4_spikes, '{0}_spikemon_i.txt'.format(areas[area_idx][0])))
            t_exc = np.loadtxt(os.path.join(fig4_spikes, '{0}_spikemon_t.txt'.format(areas[area_idx][0])))
            i_inh = np.loadtxt(os.path.join(fig4_spikes, '{0}_spikemon_i.txt'.format(areas[area_idx][1])))
            t_inh = np.loadtxt(os.path.join(fig4_spikes, '{0}_spikemon_t.txt'.format(areas[area_idx][1])))

        # fix the data
        i_exc = i_exc.astype(int)
        t_exc = t_exc*ms
        i_inh = i_inh.astype(int)
        t_inh = t_inh*ms

        # downsample the rasters
        N_exc = N_tot[3][0]
        N_inh = N_tot[3][1]

        exc_mixed = np.arange(0, N_exc, int(N_exc/N_scaling))
        inh_mixed = np.arange(0, N_inh, int(N_inh/N_scaling))

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

        # select axis
        if args.rasters_all:
            ax_curr = axs[1][area_idx]
        else:
            ax_curr = axs[1][0]

        print('[>] Plotting panel A - Rasters')
        ax_curr.scatter(t_inh_sub/ms, i_inh_sub, s=0.55, linewidth=1., marker='|', c=c_inh, edgecolors=None, alpha=1., rasterized=True)
        ax_curr.scatter(t_exc_sub/ms, i_exc_sub, s=0.55, linewidth=1., marker='|', c=c_exc, edgecolors=None, alpha=1., rasterized=True)

        # stimulation line
        ax_curr.axvline(x=t_stim, ymin=-0.5, ymax=1.1, color='gray', alpha=0.75, ls='--', linewidth=1.5, zorder=10, rasterized=False, clip_on=False)

    # Calculate the CA1 (last spikes loaded) FRs
    print('[+] Calculating CA1 FRs...')
    tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    # Normalize the FRs
    print('[+] Normalizing CA1 FRs...')
    FR_inh_norm = (FR_inh/winsize_FR)/N_tot[3][1]
    FR_exc_norm = (FR_exc/winsize_FR)/N_tot[3][0]

    print('[>] Plotting panel A - CA1 FRs')
    ax_FRs.plot(tv_inh_FR/ms, FR_inh_norm+FR_exc_norm.max()+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
    ax_FRs.plot(tv_exc_FR/ms, FR_exc_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)

    # ax_FRs.legend(loc='upper right', frameon=False, fontsize=8)
    ax_FRs.text(x=0.1, y=0.425, transform=ax_FRs.transAxes, s='Inhibitory', fontsize=fsize_xylabels, ha='center', color=c_inh, clip_on=False)
    ax_FRs.text(x=0.1, y=0.1, transform=ax_FRs.transAxes, s='Excitatory', fontsize=fsize_legends, ha='center', color=c_exc, clip_on=False)

    # stimulation line
    ax_FRs.axvline(x=t_stim, ymin=-0.5, ymax=1.1, color='gray', alpha=0.75, ls='--', linewidth=1.5, zorder=10, rasterized=False, clip_on=False)



    # Pre- vs post-stim PAC
    print('[+] Calculating pre-stim PAC...')
    tidx_pre = np.logical_and(tv_exc_FR/ms>=tlims_pre[0], tv_exc_FR/ms<=tlims_pre[1])
    tidx_post = np.logical_and(tv_exc_FR/ms>=tlims_post[0], tv_exc_FR/ms<=tlims_post[1])

    f_pha_PAC = (3, 12, 2, .2)
    f_amp_PAC = (30, 190, 20, 2)
    pac_obj = Pac(idpac=(2, 0, 0), f_pha=f_pha_PAC, f_amp=f_amp_PAC)

    # filtering
    pha_exc_pre = pac_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx_pre], ftype='phase')
    amp_exc_pre = pac_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx_pre], ftype='amplitude')
    pha_exc_post = pac_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx_post], ftype='phase')
    amp_exc_post = pac_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx_post], ftype='amplitude')

    # compute PAC
    pac_exc_pre = pac_obj.fit(pha_exc_pre, amp_exc_pre).mean(-1)
    pac_exc_post = pac_obj.fit(pha_exc_post, amp_exc_post).mean(-1)

    # plot
    vmax = np.max([pac_exc_pre.max(), pac_exc_post.max()])
    # kw = dict(vmax=vmax, vmin=.04, cmap='viridis', interp=None, plotas='imshow')
    kw = dict(vmax=vmax+0.01, vmin=.04, cmap='viridis', interp=(0.5, 0.5), plotas='imshow', fz_title=11, fz_labels=9)
    # kw = dict(vmax=vmax, vmin=.04, cmap='viridis', plotas='contour', ncontours=5)

    im_pre = ax_PAC_pre.pcolormesh(pac_exc_pre)
    im_post = ax_PAC_post.pcolormesh(pac_exc_post)

    # plt.sca(ax_PAC)
    # pac_obj.comodulogram(pac_exc, title='PAC Excitatory [-1, 0]s', colorbar=False, **kw)

    # X and Y labels
    xlbl = ax_PAC_pre.xaxis.get_label()
    ylbl = ax_PAC_pre.yaxis.get_label()

    ax_PAC_pre.set_xlabel('Phase frequency [Hz]')
    ax_PAC_pre.set_ylabel('Amplitude frequency [Hz]')


    # Tick fontsizes
    ax_PAC_pre.tick_params(axis='both', which='both', labelsize=fsize_ticks)
    ax_PAC_pre.tick_params(axis='both', which='both', labelsize=fsize_ticks)

    # plot the colorbar
    # im = plt.gca().get_children()[-2]
    # cbar_ax = fig_PAC.add_axes([0.92, 0.1, 0.01, 0.8])
    # cb = plt.colorbar(im_pre, cax=None, ax=ax_PAC_post, location='bottom', pad=0.25, ticks=[0., 0.1, 0.2, 0.3], orientation='horizontal')
    # cb.set_label('PAC values', fontsize=9, rotation='horizontal')
    #
    # cb.ax.tick_params(labelsize=9)
    # cb.outline.set_visible(False)

    # manual adjustments
    # fig_PAC.subplots_adjust(left=0.2, right = 0.95, top=0.9, bottom=0.1)

    # perfect squares
    # aspect_ratio = (f_pha_PAC[1]-f_pha_PAC[0])/(f_amp_PAC[1]-f_amp_PAC[0])
    # ax_PAC.set_aspect('auto')
    # ax_PAC.set_aspect('auto')


    # Print MI metrics
    MI_I = check_MI(FR_inh_norm)
    MI_E = check_MI(FR_exc_norm)
    print('[!] MI-I: ', MI_I)
    print('[!] MI-E: ', MI_E)

    # Save and show the figure
    print('[+] Saving the figure...')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig4', args.figure_name + '.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig4', args.figure_name + '.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

    plt.show()

    # Exit properly
    sys.exit(0)
