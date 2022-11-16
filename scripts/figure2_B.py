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
from scipy import integrate
from matplotlib import colors
from matplotlib import ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import font_manager as fm

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *
from src.figure_plots_parameters import *

from tensorpac import Pac, EventRelatedPac, PreferredPhase
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude

fontprops = fm.FontProperties(size=12, family='monospace')


# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42
mplb.rcParams['axes.titlesize'] = 11
mplb.rcParams['axes.labelsize'] = 8


def noise_test_plot(tv, fs, FR, noise, idxs, pac_obj, RND=False):
    if RND:
        noise = np.random.uniform(0, 10, FR.size)

    # Add noise
    FR_noise = FR + noise

    # Filter using PAC toolbox
    pha_nn = pac_obj.filter(fs, FR[np.newaxis, idxs], ftype='phase')
    amp_nn = pac_obj.filter(fs, FR[np.newaxis, idxs], ftype='amplitude')
    pha_noise = pac_obj.filter(fs, FR_noise[np.newaxis, idxs], ftype='phase')
    amp_noise = pac_obj.filter(fs, FR_noise[np.newaxis, idxs], ftype='amplitude')
    xval_nn = pac_obj.fit(pha_nn, amp_nn).mean(-1)
    xval_noise = pac_obj.fit(pha_noise, amp_noise).mean(-1)

    # Add MI as text
    print('Noiseless MI:', xval_nn)
    print('Noise MI:', xval_noise)

    # Make figure
    fig, axs = plt.subplots(3,1,figsize=(12,10))

    # Plot
    axs[0].plot(tv[idxs], FR_noise[idxs], label='noisy')
    axs[0].plot(tv[idxs], FR[idxs]+150, label='noiseless')
    axs[1].plot(tv[idxs], pha_noise.squeeze())
    axs[1].plot(tv[idxs], pha_nn.squeeze())
    axs[2].plot(tv[idxs], amp_noise.squeeze())
    axs[2].plot(tv[idxs], amp_nn.squeeze())

    # Add MI as text
    print(xval_noise)
    print(xval_nn)
    axs[0].text(x=0.42, y=0.25, s='MI = {0}'.format(xval_noise.squeeze()), fontsize=10, ha='center', va='bottom', color='C0', transform=axs[0].transAxes)
    axs[0].text(x=0.42, y=0.75, s='MI = {0}'.format(xval_nn.squeeze()), fontsize=10, ha='center', va='bottom', color='C1', transform=axs[0].transAxes)

    # Show legend
    axs[0].legend(loc=0)

    # Pretty stuff
    axs[0].set_title('Firing Rates')
    axs[1].set_title('Phase')
    axs[2].set_title('Amplitude')
    axs[2].set_xlabel('Time [s]')
    axs[0].set_ylabel('FRs [Hz]')
    axs[1].set_ylabel('Phase [rad]')
    axs[2].set_ylabel('Amplitude')

    # apply tight layout
    fig.set_tight_layout(True)

    return fig,axs


def check_MI_noisy(tv, fs, FR, noise, idxs, RND=False):
    if RND:
        noise = np.random.uniform(0, 10, FR.size)

    # Add noise
    FR_noise = FR + noise

    # Define frequency ranges
    f_pha = [3, 9]
    f_amp0 = [20, 40]
    f_amp1 = [40, 80]

    # Make PAC objects
    pac_obj0 = Pac(idpac=(2,0,0), f_pha=f_pha, f_amp=f_amp0)
    pac_obj1 = Pac(idpac=(2,0,0), f_pha=f_pha, f_amp=f_amp1)

    # Filter and fit
    pha0 = pac_obj0.filter(fs, FR_noise[np.newaxis, idxs], ftype='phase')
    amp0 = pac_obj0.filter(fs, FR_noise[np.newaxis, idxs], ftype='amplitude')
    pha1 = pac_obj1.filter(fs, FR_noise[np.newaxis, idxs], ftype='phase')
    amp1 = pac_obj1.filter(fs, FR_noise[np.newaxis, idxs], ftype='amplitude')
    xval0 = pac_obj0.fit(pha0, amp0).mean(-1)
    xval1 = pac_obj1.fit(pha1, amp1).mean(-1)

    # Make the figures
    fig, axs = plt.subplots(3, 1, figsize=(12,10))

    # Plot the lines
    axs[0].plot(tv[idxs], FR_noise[idxs], label='noisy')
    axs[0].plot(tv[idxs], FR[idxs]+150, label='noiseless')
    axs[1].plot(tv[idxs], pha0.squeeze(), label='[20,40]')
    axs[1].plot(tv[idxs], pha1.squeeze(), label='[40,80]')
    axs[2].plot(tv[idxs], amp0.squeeze())
    axs[2].plot(tv[idxs], amp1.squeeze())

    # Add MI as text
    print(f_amp0, xval0)
    print(f_amp1, xval1)
    axs[2].text(x=0.42, y=0.65, s='MI = {0}'.format(xval0.squeeze()), fontsize=10, ha='center', va='bottom', color='C0', transform=axs[2].transAxes)
    axs[2].text(x=0.42, y=0.75, s='MI = {0}'.format(xval1.squeeze()), fontsize=10, ha='center', va='bottom', color='C1', transform=axs[2].transAxes)

    # Pretty plots
    axs[0].set_title('Firing Rates')
    axs[1].set_title('Phases')
    axs[2].set_title('Amplitudes')
    axs[2].set_xlabel('Time [s]')
    axs[0].set_ylabel('FR [Hz]')
    axs[1].set_ylabel('Phases [rad]')
    axs[2].set_ylabel('Amplitudes')

    # Show legends
    axs[0].legend(loc=0)
    axs[1].legend(loc=0)

    # apply tight layout
    fig.set_tight_layout(True)

    return fig, axs


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

    # Define delta lower and upper limits
    theta_band = [3, 10]
    gamma_low_band = [40, 80]
    gamma_high_band = [80, 120]

    # Bands for PAC / PreferredPhase
    f_pha_PAC = (3, 12, 2, .2)
    f_amp_PAC = (30, 100, 10, 1)
    f_pha_PP = [3, 9]
    f_amp_PP = (20, 100, 10, 1)
    f_pha_MI = [3, 9]
    f_amp_MI = [40, 80]

    # Figure sizes
    fig_width = 2.3
    fig_height = 2.6

    # Font size
    fsize = 9

    # Normalize the FRs
    print('[+] Normalizing the FRs...')
    noise = np.random.uniform(0, 10, FR_exc.size)
    FR_inh_norm = (FR_inh/winsize_FR)/N_CA1_inh
    FR_exc_norm = (FR_exc/winsize_FR)/N_CA1_exc
    # FR_inh_norm += noise
    # FR_exc_norm += noise


    # PSD calculations
    print('[+] Calculating PSDs...')
    PSD_args = {'detrend' : 'constant',
                'return_onesided' : True,
                'scaling' : 'density'}
    fv, PSD_inh = my_PSD(FR_inh_norm, fs_FR, winsize_samples, noverlap, k=6, **PSD_args)
    _, PSD_exc = my_PSD(FR_exc_norm, fs_FR, winsize_samples, noverlap, k=6, **PSD_args)

    # Use the TensorPAC module to calculate the PSDs
    psd_pac_inh = PSD(FR_inh_norm[np.newaxis,tidx], fs_FR)
    psd_pac_exc = PSD(FR_exc_norm[np.newaxis,tidx], fs_FR)

    # Calculate peaks
    peaks_inh, props_inh = sig.find_peaks(psd_pac_inh.psd[0], prominence=1, threshold=psd_pac_inh.psd[0].max()*0.2)
    peaks_exc, props_exc = sig.find_peaks(psd_pac_exc.psd[0], prominence=1, threshold=psd_pac_exc.psd[0].max()*0.2)

    # Find intersecting values in frequency vector
    idx_theta = np.logical_and(psd_pac_inh.freqs >= theta_band[0], psd_pac_inh.freqs <= theta_band[1])
    idx_gamma_low = np.logical_and(psd_pac_inh.freqs >= gamma_low_band[0], psd_pac_inh.freqs <= gamma_low_band[1])
    idx_gamma_high = np.logical_and(psd_pac_inh.freqs >= gamma_high_band[0], psd_pac_inh.freqs <= gamma_high_band[1])

    # Calculate the power in theta/gamma bands
    # Compute the absolute power by approximating the area under the curve
    df = fv[1] - fv[0]
    theta_power_inh = integrate.simpson(psd_pac_inh.psd[0][idx_theta], dx=df)
    low_gamma_power_inh = integrate.simpson(psd_pac_inh.psd[0][idx_gamma_low], dx=df)
    theta_power_exc = integrate.simpson(psd_pac_exc.psd[0][idx_theta], dx=df)
    low_gamma_power_exc = integrate.simpson(psd_pac_exc.psd[0][idx_gamma_low], dx=df)

    print('*'+'='*32)
    print('Absolute theta power [I]: {:.3f} uV^2'.format(theta_power_inh))
    print('Absolute theta power [E]: {:.3f} uV^2'.format(theta_power_exc))
    print('Absolute low-gamma power [I]: {:.3f} uV^2'.format(low_gamma_power_inh))
    print('Absolute low-gamma power [E]: {:.3f} uV^2'.format(low_gamma_power_exc))
    print('*'+'='*32)
    print('Relative theta power [I]: {:.3%}'.format(theta_power_inh /integrate.simpson(psd_pac_inh.psd[0], dx=df)))
    print('Relative theta power [E]: {:.3%}'.format(theta_power_exc / integrate.simpson(psd_pac_exc.psd[0], dx=df)))
    print('Relative low-gamma power [I]: {:.3%}'.format(low_gamma_power_inh/integrate.simpson(psd_pac_inh.psd[0], dx=df)))
    print('Relative low-gamma power [E]: {:.3%}'.format(low_gamma_power_exc/integrate.simpson(psd_pac_exc.psd[0], dx=df)))
    print('='*32+'*')


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
    ax_PSD.tick_params(axis='both', which='both', labelsize=fsize_ticks)
    axin_PSD.tick_params(axis='both', which='both', labelsize=fsize_ticks)

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
    axin_PSD.tick_params(axis='y', which='both', labelsize=fsize_ticks, length=0, width=0)

    # Actually plotting
    ax_PSD.plot(psd_pac_inh.freqs, psd_pac_inh.psd[0], c=c_inh, label='Inhibitory', zorder=1, rasterized=False)
    ax_PSD.plot(psd_pac_exc.freqs, psd_pac_exc.psd[0], c=c_exc, label='Excitatory', zorder=2, rasterized=False)

    # Fill areas below curve
    # ax_PSD.fill_between(psd_pac_inh.freqs, psd_pac_inh.psd[0], where=idx_theta, color=c_inh, alpha=0.2)
    # ax_PSD.fill_between(psd_pac_exc.freqs, psd_pac_exc.psd[0], where=idx_theta, color=c_exc, linestyle='--', linewidth=1.0, alpha=0.2)

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
    ax_PSD.legend(fontsize=fsize_legends, frameon=False)

    # Titles
    ax_PSD.set_title('CA1 Spectral Analysis')

    # Add the lines to indicate where the inset axis is coming from
    ax_PSD.indicate_inset_zoom(axin_PSD)
    # ax_PSD.indicate_inset_zoom(axin_PSD, edgecolor="black")


    # PAC test
    # Modulation Index
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
    ax_PAC.tick_params(axis='both', which='both', labelsize=fsize_ticks)

    # plot the colorbar
    im = plt.gca().get_children()[-2]
    # cbar_ax = fig_PAC.add_axes([0.92, 0.1, 0.01, 0.8])
    cb = plt.colorbar(im, cax=None, ax=ax_PAC, location='bottom', pad=0.25, ticks=[0., 0.1, 0.2, 0.3], orientation='horizontal')
    cb.set_label('PAC values', fontsize=fsize_xylabels, rotation='horizontal')

    cb.ax.tick_params(labelsize=fsize_ticks)
    cb.outline.set_visible(False)

    # manual adjustments
    # fig_PAC.subplots_adjust(left=0.2, right = 0.95, top=0.9, bottom=0.1)

    # perfect squares
    # aspect_ratio = (f_pha_PAC[1]-f_pha_PAC[0])/(f_amp_PAC[1]-f_amp_PAC[0])
    # ax_PAC.set_aspect('auto')
    # ax_PAC.set_aspect('auto')


    # bins plot (sanity check)
    # define phase and amplitude filtering properties
    kw_filt = dict(f_pha=[4, 12], f_amp=(30, 140, 20, 2), n_bins=36)
    bin_exc = BinAmplitude(FR_exc_norm, fs_FR, **kw_filt)
    bin_inh = BinAmplitude(FR_inh_norm, fs_FR, **kw_filt)

    plt.figure('bins')
    plt.subplot(2,1,1)
    bin_exc.plot(normalize=True, color='blue', unit='deg')
    plt.title('Excitatory', fontsize=fsize_titles)
    plt.subplot(2,1,2)
    bin_inh.plot(normalize=True, color='red', unit='deg')
    plt.title('Inhibitory', fontsize=fsize_titles)
    # plt.tight_layout()


    # PreferredPhase plot (sanity check #2)
    # pp_obj = PreferredPhase(f_pha=[3, 9], f_amp=(20, 100, 10, 1))
    pp_obj = PreferredPhase(f_pha=f_pha_PP, f_amp=f_amp_PP)
    pac_obj2 = Pac(idpac=(2, 0, 0), f_pha=f_pha_MI, f_amp=f_amp_MI)
    # pp_obj = PreferredPhase(f_pha=f_pha_PAC, f_amp=f_amp_PAC)
    # pp_obj = PreferredPhase(f_pha=(3, 12, 2, .2), f_amp=(30, 120, 20, 2))
    pp_pha_exc = pp_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx], ftype='phase')
    pp_amp_exc = pp_obj.filter(fs_FR, FR_exc_norm[np.newaxis,tidx], ftype='amplitude')
    pp_pha_inh = pp_obj.filter(fs_FR, FR_inh_norm[np.newaxis,tidx], ftype='phase')
    pp_amp_inh = pp_obj.filter(fs_FR, FR_inh_norm[np.newaxis,tidx], ftype='amplitude')

    # compute the preferred phase (can reuse the amplitude computed above)
    # Now, compute the PP :
    ampbin_inh, pp_inh, polarvec_inh = pp_obj.fit(pp_pha_inh, pp_amp_inh, n_bins=72)
    ampbin_exc, pp_exc, polarvec_exc = pp_obj.fit(pp_pha_exc, pp_amp_exc, n_bins=72)

    # compute the MI in the [40, 80] Hz band
    xpac = pac_obj2.t(fs_FR, FR_exc_norm[np.newaxis,tidx])

    # start plotting
    plt.figure('pref_phase')
    # plt.sca(ax_prefp)
    # kw_plt = dict(cmap='Spectral_r', interp=.1, cblabel='Amplitude bins',
                  # vmin=0.012, vmax=0.03, y=1.05, fz_title=18)
    kw_plt = dict(cmap='viridis', interp=.1, cblabel='Amplitude bins',
                  vmin=0.012, vmax=0.03, y=1., fz_title=11, fz_labels=9)
    # ax1 = pp_obj.polar(ampbin_inh.squeeze().T, vecbin, pp_obj.yvec, subplot=121,
    #              title='Inhibitory', colorbar=False, **kw_plt)
    ax_prefp = pp_obj.polar(ampbin_exc.squeeze().T, polarvec_exc, pp_obj.yvec, title='Phase-Amplitude Coupling', colorbar=False, **kw_plt)
    ax_prefp.yaxis.grid(linewidth=0.5, alpha=0.9)

    # Create the patches
    # p_low = mplb.patches.Rectangle((np.deg2rad(0), 50), np.deg2rad(360), 20, color=None, alpha=0.3, facecolor='white', edgecolor='red', linewidth=1)
    # p_high = mplb.patches.Rectangle((np.deg2rad(0), 110), np.deg2rad(360), 20, color=None, alpha=0.25, facecolor='grey', edgecolor='red', linewidth=1)
    # p_low = mplb.patches.Rectangle((np.deg2rad(0), 30), np.deg2rad(360), 10, color=None, alpha=0.3, facecolor='black', edgecolor='white', linewidth=2)
    # p_high = mplb.patches.Rectangle((np.deg2rad(0), 80), np.deg2rad(360), 20, color=None, alpha=0.3, facecolor='black', edgecolor='white', linewidth=2)
    # p_show = mplb.patches.Rectangle((np.deg2rad(0), 40), np.deg2rad(360), 40, color=None, alpha=1., facecolor='none', edgecolor='white', linewidth=2)

    # Add them on the figure
    # ax_prefp.add_patch(p_low)
    # ax_prefp.add_patch(p_high)
    # ax_prefp.add_patch(p_show)

    # add MI as a text
    # mi_text = r'MI [3,7] vs [40,80]: {0:.2f}'.format(xpac.squeeze())
    # mi_text = r'  MI: {0:.4f} // {1:.4E}'.format(xpac.squeeze(), xpac.squeeze())
    mi_text = r'  MI: {0:.4f}'.format(xpac.squeeze())
    ax_prefp.text(x=np.deg2rad(-90), y=70, s=mi_text, fontsize=fsize_misc, ha='center', va='bottom', color='white', clip_on=True)

    # Fix the label positions
    ax_prefp.set_rlabel_position(135)
    # ax_prefp.set_rlabel_position(90)
    for label in ax_prefp.get_ymajorticklabels():
        label.set_color('white')

    # Title
    # ax_prefp.set_title('Excitatory', fontsize=fsize_titles)

    # Set the ticks
    ax_prefp.xaxis.set_ticklabels(['', '', '', r'$\theta_{peak}$', '', '', '', r'$\theta_{trough}$'])
    ax_prefp.set_yticks([40, 60, 80, 100])
    ax_prefp.yaxis.set_ticklabels(['40 Hz', '60 Hz', '80 Hz', '120 Hz'])
    ax_prefp.tick_params(axis='both', which='both', labelsize=fsize_ticks)
    ax_prefp.tick_params(axis='x', which='both', rotation=90, pad=-0.5, labelsize=fsize_misc)
    ax_prefp.set_thetalim(-np.pi, np.pi)

    # Set the xy-lims
    # ax_prefp.set_xlim(37, 81)
    ax_prefp.set_ylim(30, 90)

    # Set origins
    ax_prefp.set_rorigin(10)
    # ax_prefp.set_theta_zero_location('W', offset=10)

    # plot the colorbar
    # im = plt.gca().get_children()[0]
    im = ax_prefp.collections[-1]
    # fig_prefp.subplots_adjust(left=0.05, right = 0.9, top=0.9, bottom=0.1)
    # cbar_ax = fig_prefp.add_axes([0.85, 0.1, 0.05, 0.8])
    # cbar_ax = fig_prefp.add_axes([0.05, 0.05, 0.9, 0.1])
    cb = plt.colorbar(im, cax=None, ax=ax_prefp, pad=0.05, location='bottom', orientation='horizontal')
    # cb.ax.set_yticks([0.012, 0.03])
    # cb.ax.set_yticklabels(['0', '1'])
    clims = im.get_clim()
    cb.set_ticks([min(clims), max(clims)])
    cb.set_ticklabels(['min', 'max'])
    cb.ax.tick_params(labelsize=fsize_ticks)
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
