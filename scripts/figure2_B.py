#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### PSD and PAC!

import os
import sys
from pathlib import Path

import json

import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt
from scipy import signal as sig
from matplotlib import colors
from matplotlib import ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import font_manager as fm

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *


from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude


fontprops = fm.FontProperties(size=12, family='monospace')

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42
mplb.rcParams['axes.titlesize'] = 11
mplb.rcParams['axes.labelsize'] = 8

# main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Figure_2B from paper')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig2_A',
                        help='Name of the output figure [without extension].')

    args = parser.parse_args()


    """ Loading the FRs from disk """
    print('[+] Loading FRs from disk...')

    # Load the non-normalized FRs
    FR_inh = np.loadtxt(os.path.join(parent_dir, 'test_FRs', 'FR_inh.txt'))
    tv_FR_inh = np.loadtxt(os.path.join(parent_dir, 'test_FRs', 'tv_inh.txt'))
    FR_exc = np.loadtxt(os.path.join(parent_dir, 'test_FRs', 'FR_exc.txt'))
    tv_FR_exc = np.loadtxt(os.path.join(parent_dir, 'test_FRs', 'tv_exc.txt'))

    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Area names and sizes
    areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
    area_labels = ['EC', 'DG', 'CA3', 'CA1']
    N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]
    N_CA1_exc = N_tot[3][0]
    N_CA1_inh = N_tot[3][1]

    # Timing
    second = 1
    ms = 1e-3
    duration = 3*second
    dt = 0.1*ms
    fs = int(1/dt)
    t_lims = [950*ms, 2700*ms] # ms
    t_lims_adj = [1000*ms, 2000*ms]
    tidx = np.logical_and(tv_FR_inh>=t_lims_adj[0], tv_FR_inh<=t_lims_adj[1])

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'

    # Calculate the sampling frequency of the FRs
    fs_FR = int(1/(tv_FR_exc[2] - tv_FR_exc[1]))

    # From fig2_A:
    winsize_FR = 5*ms

    # PSDs
    overlap_PSD = 0.9
    winsize_PSD = 1000*ms
    winstep_PSD = winsize_PSD*round(1-overlap_PSD,4)

    winsize_samples = int(winsize_PSD*fs_FR)
    winstep_samples = int(winstep_PSD*fs_FR)
    noverlap = int(overlap_PSD*winsize_samples)

    # Figure sizes
    fig_width = 2.3
    fig_height = 2.6

    # Font size
    fsize = 9

    # Normalize the FRs
    print('[+] Normalizing the FRs...')
    FR_inh_norm = (FR_inh/winsize_FR)/N_CA1_inh
    FR_exc_norm = (FR_exc/winsize_FR)/N_CA1_exc

    # PSD calculations
    print('[+] Calculating PSDs...')
    PSD_args = {'detrend' : 'constant',
                'return_onesided' : True,
                'scaling' : 'density'}
    fv, PSD_inh = my_PSD(FR_inh_norm, fs_FR, winsize_samples, noverlap, k=4, **PSD_args)
    _, PSD_exc = my_PSD(FR_exc_norm, fs_FR, winsize_samples, noverlap, k=4, **PSD_args)

    # Use the TensorPAC module to calculate the PSDs
    psd_pac_inh = PSD(FR_inh_norm[np.newaxis,tidx], fs_FR)
    psd_pac_exc = PSD(FR_exc_norm[np.newaxis,tidx], fs_FR)


    # Plot the data
    #------------------------
    print('[>] Plotting the PSDs...')

    # Make the figures
    #------------------------
    print('[+] Making the figures...')
    fig_PSD = plt.figure('PSD', figsize=(fig_width, fig_height))
    fig_PAC, ax_PAC = plt.subplots(1, 1, num='PAC', figsize=(fig_width, fig_height))
    fig_bins = plt.figure('bins', figsize=(fig_width, fig_height))
    fig_prefp = plt.figure('pref_phase', figsize=(fig_width, fig_height))
    # fig_prefp, ax_prefp = plt.subplots(1, 3, num='pref_phase', figsize=(18,8), gridspec_kw={'width_ratios': [0.475, 0.475, 0.05]})

    # Make the axes
    #------------------------
    print('[+] Making the axes...')
    # ax0_PSD = fig_PSD.add_subplot(211)
    # ax1_PSD = fig_PSD.add_subplot(212)
    # ax_PSD = fig_PSD.add_axes([0.05, 0.05, 0.9, 0.9])
    ax_PSD = fig_PSD.add_subplot()
    axin_PSD = ax_PSD.inset_axes([0.3, 0.3, 0.65, 0.4])

    # Hide some spines
    ax_PSD.spines['top'].set_visible(False)
    # ax_PSD.spines['bottom'].set_visible(False)
    # ax_PSD.spines['left'].set_visible(False)
    ax_PSD.spines['right'].set_visible(False)

    # Color the spines of the inset
    for spine in axin_PSD.spines.values():
        spine.set_edgecolor('#cdcdcd')

    # Labels
    # ax_PSD.set_ylabel('PSD [$\\frac{V^2}{Hz}$]', labelpad=-25.)
    ax_PSD.set_ylabel('PSD', labelpad=-15.)
    ax_PSD.set_xlabel('Frequency [Hz]')

    # Tick fontsizes
    ax_PSD.tick_params(axis='both', which='both', labelsize=9)
    axin_PSD.tick_params(axis='both', which='both', labelsize=9)

    # Ticks of yaxis rotated
    # for tick in ax_PSD.get_yticklabels():
    #     tick.set_rotation(-90)

    # Remove some ticks
    ax_PSD.yaxis.set_major_locator(ticker.FixedLocator([0, 1000, 2000, 3000]))
    ax_PSD.yaxis.set_minor_locator(ticker.FixedLocator([500, 1500, 2500]))
    ax_PSD.yaxis.set_ticklabels(['0', '', '', 'Max'])
    axin_PSD.xaxis.set_minor_locator(ticker.FixedLocator([50, 70]))
    axin_PSD.yaxis.set_major_locator(ticker.FixedLocator([0, 50]))
    axin_PSD.yaxis.set_minor_locator(ticker.FixedLocator([25]))
    axin_PSD.yaxis.set_ticklabels(['', ''])
    axin_PSD.tick_params(axis='y', which='both', labelsize=9, length=0, width=0)


    # Actually plotting
    ax_PSD.plot(psd_pac_inh.freqs, psd_pac_inh.psd[0], c=c_inh, label='Inhibitory', zorder=1, rasterized=False)
    ax_PSD.plot(psd_pac_exc.freqs, psd_pac_exc.psd[0], c=c_exc, label='Excitatory', zorder=2, rasterized=False)

    # inset
    axin_PSD.plot(psd_pac_inh.freqs, psd_pac_inh.psd[0], c=c_inh, label='Inhibitory', zorder=1, rasterized=False)
    axin_PSD.plot(psd_pac_exc.freqs, psd_pac_exc.psd[0], c=c_exc, label='Excitatory', zorder=2, rasterized=False)

    # Setting xlims
    ax_PSD.set_xlim([0, 120])
    axin_PSD.set_xlim(37, 81)
    axin_PSD.set_ylim(0, 60)

    # gridlines
    axin_PSD.grid(which='major', alpha=0.5, linestyle='dotted', linewidth=1)

    # Legend
    ax_PSD.legend(fontsize=8, frameon=False)

    # Titles
    ax_PSD.set_title('CA1 Spectral Analysis')

    # Add the lines to indicate where the inset axis is coming from
    ax_PSD.indicate_inset_zoom(axin_PSD)
    # ax_PSD.indicate_inset_zoom(axin_PSD, edgecolor="black")


    # PAC test
    # Modulation Index
    f_pha_PAC = (3, 12, 2, .2)
    f_amp_PAC = (30, 190, 20, 2)
    pac_obj = Pac(idpac=(2, 0, 0), f_pha=f_pha_PAC, f_amp=f_amp_PAC)
    # pac_obj = Pac(idpac=(2, 0, 0), f_pha=np.arange(4,13,2), f_amp=np.arange(30,121,2))
    # pac_obj = Pac(idpac=(2, 0, 0), f_pha='hres', f_amp='mres')

    # filtering
    pha_p_inh = pac_obj.filter(fs_FR, FR_inh_norm[np.newaxis,tidx], ftype='phase')
    amp_p_inh = pac_obj.filter(fs_FR, FR_inh_norm[np.newaxis,tidx], ftype='amplitude')
    pha_p_exc = pac_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx], ftype='phase')
    amp_p_exc = pac_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx], ftype='amplitude')

    # compute PAC
    pac_inh = pac_obj.fit(pha_p_inh, amp_p_inh).mean(-1)
    pac_exc = pac_obj.fit(pha_p_exc, amp_p_exc).mean(-1)

    # plot
    vmax = np.max([pac_inh.max(), pac_exc.max()])
    # kw = dict(vmax=vmax, vmin=.04, cmap='viridis', interp=None, plotas='imshow')
    kw = dict(vmax=vmax+0.01, vmin=.04, cmap='viridis', interp=(0.5, 0.5), plotas='imshow', fz_title=11, fz_labels=9)
    # kw = dict(vmax=vmax, vmin=.04, cmap='viridis', plotas='contour', ncontours=5)
    plt.figure('PAC')
    # plt.subplot(121)
    # ax1 = pac_obj.comodulogram(pac_inh, title='PAC Inhibitory [-1, 10]s', colorbar=False, **kw)
    # plt.pcolormesh(pac_inh)
    # plt.subplot(111)
    # fig_PAC.add_subplots()
    plt.sca(ax_PAC)
    pac_obj.comodulogram(pac_exc, title='PAC Excitatory [-1, 0]s', colorbar=False, **kw)

    # X and Y labels
    xlbl = ax_PAC.xaxis.get_label()
    ylbl = ax_PAC.yaxis.get_label()

    ax_PAC.set_xlabel('Phase frequency [Hz]')
    ax_PAC.set_ylabel('Amplitude frequency [Hz]')


    # Tick fontsizes
    ax_PAC.tick_params(axis='both', which='both', labelsize=9)

    # plot the colorbar
    im = plt.gca().get_children()[-2]
    # cbar_ax = fig_PAC.add_axes([0.92, 0.1, 0.01, 0.8])
    cb = plt.colorbar(im, cax=None, ax=ax_PAC, location='bottom', pad=0.25, ticks=[0., 0.1, 0.2, 0.3], orientation='horizontal')
    cb.set_label('PAC values', fontsize=9, rotation='horizontal')

    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)

    # manual adjustments
    # fig_PAC.subplots_adjust(left=0.2, right = 0.95, top=0.9, bottom=0.1)

    # perfect squares
    # aspect_ratio = (f_pha_PAC[1]-f_pha_PAC[0])/(f_amp_PAC[1]-f_amp_PAC[0])
    # ax_PAC.set_aspect('auto')
    # ax_PAC.set_aspect('auto')


    # bins plot (sanity check)
    # define phase and amplitude filtering properties
    kw_filt = dict(f_pha=[4, 12], f_amp=(30, 120, 20, 2), n_bins=36)
    bin_exc = BinAmplitude(FR_exc_norm, fs_FR, **kw_filt)
    bin_inh = BinAmplitude(FR_inh_norm, fs_FR, **kw_filt)

    plt.figure('bins')
    plt.subplot(2,1,1)
    bin_exc.plot(normalize=True, color='blue', unit='deg')
    plt.title('Excitatory', fontsize=18)
    plt.subplot(2,1,2)
    bin_inh.plot(normalize=True, color='red', unit='deg')
    plt.title('Inhibitory', fontsize=18)
    # plt.tight_layout()


    # PreferredPhase plot (sanity check #2)
    pp_obj = PreferredPhase(f_pha=[4, 12], f_amp=(10, 140, 20, 2))
    # pp_obj = PreferredPhase(f_pha=(3, 12, 2, .2), f_amp=(30, 120, 20, 2))
    pp_pha_exc = pp_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx], ftype='phase')
    pp_amp_exc = pp_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx], ftype='amplitude')
    pp_pha_inh = pp_obj.filter(fs_FR, FR_inh_norm[np.newaxis,tidx], ftype='phase')
    pp_amp_inh = pp_obj.filter(fs_FR, FR_inh_norm[np.newaxis,tidx], ftype='amplitude')

    # compute the preferred phase (can reuse the amplitude computed above)
    ampbin_exc, pp_exc, vecbin = pp_obj.fit(pp_pha_exc, pp_amp_exc, n_bins=72)
    ampbin_inh, pp_inh, vecbin = pp_obj.fit(pp_pha_inh, pp_amp_inh, n_bins=72)

    plt.figure('pref_phase')
    # plt.sca(ax_prefp)
    # kw_plt = dict(cmap='Spectral_r', interp=.1, cblabel='Amplitude bins',
                  # vmin=0.012, vmax=0.03, y=1.05, fz_title=18)
    kw_plt = dict(cmap='viridis', interp=.1, cblabel='Amplitude bins',
                  vmin=0.012, vmax=0.03, y=1., fz_title=11, fz_labels=9)
    # ax1 = pp_obj.polar(ampbin_inh.squeeze().T, vecbin, pp_obj.yvec, subplot=121,
    #              title='Inhibitory', colorbar=False, **kw_plt)
    ax_prefp = pp_obj.polar(ampbin_exc.squeeze().T, vecbin, pp_obj.yvec, title='Phase-Amplitude Coupling', colorbar=False, **kw_plt)

    ax_prefp.set_rlabel_position(135)
    # ax_prefp.set_rlabel_position(90)
    for label in ax_prefp.get_ymajorticklabels():
        label.set_color('white')

    # Title
    # ax_prefp.set_title('Excitatory', fontsize=11)

    # Set the ticks
    ax_prefp.xaxis.set_ticklabels(['', '', '', r'$\theta_{peak}$', '', '', '', r'$\theta_{trough}$'])
    ax_prefp.set_yticks([60, 120])
    ax_prefp.yaxis.set_ticklabels(['60 Hz', '120 Hz'])
    ax_prefp.tick_params(axis='both', which='both', labelsize=9)
    ax_prefp.tick_params(axis='x', which='both', rotation=90, pad=-0.5)
    ax_prefp.set_thetalim(-np.pi, np.pi)

    # plot the colorbar
    # im = plt.gca().get_children()[0]
    im = ax_prefp.collections[-1]
    # fig_prefp.subplots_adjust(left=0.05, right = 0.9, top=0.9, bottom=0.1)
    # cbar_ax = fig_prefp.add_axes([0.85, 0.1, 0.05, 0.8])
    # cbar_ax = fig_prefp.add_axes([0.05, 0.05, 0.9, 0.1])
    cb = plt.colorbar(im, cax=None, ax=ax_prefp, pad=0.05, location='bottom', orientation='horizontal')
    # cb.ax.set_yticks([0.012, 0.03])
    # cb.ax.set_yticklabels(['0', '1'])
    cb.set_ticks([0.012, 0.03])
    cb.set_ticklabels([0, 1])
    # cb.set_label('Amplitude bins', fontsize=9, rotation='horizontal')
    cb.set_label('')

    cb.outline.set_visible(False)

    # adjust positions
    # ax1.set_position([0.05, 0.05, 0.95, 0.95])
    # ax2.set_position(pos=[0.55, 0.05, 0.95, 0.95])

    plt.subplots_adjust(left=0.1, right = 0.9, top=0.95, bottom=0.05)
    # plt.subplots_adjust(wspace=0.9, hspace=0.9, right=0.7)

    # perfect circles
    ax_prefp.set_aspect(1)


    # plt.tight_layout()

    # Adjustting the figures
    print('[>] Adjusting the figure dimensions...')
    fig_PSD.subplots_adjust(left=0.17, bottom=0.15, top=0.915, right=0.99)
    fig_PAC.subplots_adjust(left=0.24, bottom=0.06, top=0.89, right=0.96)
    fig_prefp.subplots_adjust(left=0.11, bottom=0., top=1.0, right=0.90)


    # Saving the figures
    print('[>] Saving the figure panels...')

    # Select PSD figure
    print('[+] Saving PSDs...')
    plt.figure('PSD')
    # plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'figures', 'fig2', 'fig2_B_PSD.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    plt.savefig(os.path.join(parent_dir, 'figures', 'fig2', 'fig2_B_PSD.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

    # Select the PAC figure
    print('[+] Saving PACs...')
    plt.figure('PAC')
    # plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'figures', 'fig2', 'fig2_X_PAC.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    plt.savefig(os.path.join(parent_dir, 'figures', 'fig2', 'fig2_X_PAC.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

    # Select the binned PAC figure
    print('[+] Saving bins...')
    plt.figure('bins')
    # plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'figures', 'fig2', 'fig2_X_bins.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    plt.savefig(os.path.join(parent_dir, 'figures', 'fig2', 'fig2_X_bins.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

    # Select the preferred phase polar figure
    print('[+] Saving preferred phase polar plot...')
    plt.figure('pref_phase')
    # plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'figures', 'fig2', 'fig2_C_prefp.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    plt.savefig(os.path.join(parent_dir, 'figures', 'fig2', 'fig2_C_prefp.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

    # Close some figures
    print('[-] Closing figure [binned phases]...')
    plt.close(fig_bins)

    # print('[-] Closing figure [polar preferred phases]...')
    # plt.close(fig_prefp)

    # Show the figures
    plt.show()

    print('[!] Done')
    exit(0)
