#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal as sig
from matplotlib import colors
from matplotlib import ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

fontprops = fm.FontProperties(size=12, family='monospace')


def my_FR(spikes: np.ndarray,
            duration: int,
            window_size: float,
            overlap: float) -> (np.ndarray, np.ndarray):
    """
    Compute the firing rate using a windowed moving average.

    Parameters
    ----------
    spikes: numpy.ndarray
        The spike times (Brian2 format, in ms)
    duration: int
        The duration of the recording (in seconds)
    window_size: float
        Width of the moving average window (in seconds)
    overlap: float
        Desired overlap between the windows (percentage, in [0., 1.))

    Returns
    -------
    t: numpy.ndarray
        Array of time values for the computed firing rate. These are the window centers.
    FR: numpy.ndarray
        Spikes per window (needs to be normalized)
    """

    # Calculate new sampling times
    win_step = window_size * round(1. - overlap, 4)
    # fs_n = int(1/win_step)

    # First center is at the middle of the first window
    c0 = window_size/2
    cN = duration-c0

    # centers
    centers = np.arange(c0, cN+win_step, win_step)

    # Calculate windowed FR
    counts = []
    for center in centers:
        cl = center - c0
        ch = center + c0
        spike_cnt = np.count_nonzero(np.where((spikes >= cl) & (spikes < ch)))
        counts.append(spike_cnt)

    # return centers and spike counts per window
    return centers, np.array(counts)


def my_specgram(signal: np.ndarray,
                   fs: int,
                   window_width: int,
                   window_overlap: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Computes the power spectrum of the specified signal.

    A periodic Hann window with the specified width and overlap is used.

    Parameters
    ----------
    signal: numpy.ndarray
        The input signal
    fs: int
        Sampling frequency of the input signal
    window_width: int
        Width of the Hann windows in samples
    window_overlap: int
        Overlap between Hann windows in samples

    Returns
    -------
    f: numpy.ndarray
        Array of frequency values for the first axis of the returned spectrogram
    t: numpy.ndarray
        Array of time values for the second axis of the returned spectrogram
    sxx: numpy.ndarray
        Power spectrogram of the input signal with axes [frequency, time]
    """
    f, t, Sxx = sig.spectrogram(x=signal,
                                # nfft=2048,
                                detrend=False,
                                fs=fs,
                                window=sig.windows.hann(M=window_width, sym=False),
                                nperseg=window_width,
                                noverlap=window_overlap,
                                return_onesided=True,
                                # scaling='spectrum',
                                mode='magnitude')

    return f, t, (1.0 / window_width) * (Sxx ** 2)


def add_sizebar(ax, xlocs, ylocs, bcolor, text):
    """ Add a sizebar to the provided axis """
    ax.plot(xlocs, ylocs, ls='-', c=bcolor, linewidth=1., rasterized=True, clip_on=False)
    ax.text(x=xlocs[0]+10*ms, y=ylocs[0], s=text, va='center', ha='left', clip_on=False)


def myround(x, base=100):
    """ Rounding function to closest *base* """
    val = int(base * round(x/base))
    return val if val > 0 else base*1


# main program
if __name__ == "__main__":

    """ Parameters initialization """
    print('[+] Setting up parameters...')
    # Timing
    second = 1
    ms = 1e-3
    duration = 3*second
    dt = 0.1*ms
    fs = int(1/dt)
    winsize_FR = 5*ms
    overlap_FR = 0.9
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)
    binnum = int(duration/winsize_FR)
    t_stim = 1715*ms
    t_lims = [0*ms, 3050*ms] # ms
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
    newcmp_inh = ListedColormap(cvals_inh)

    # excitatory cmap
    reds = np.linspace(1., c_exc_RGB[0], N)
    greens = np.linspace(1., c_exc_RGB[1], N)
    blues = np.linspace(1., c_exc_RGB[2], N)
    cvals_exc[:,0] = reds[...]
    cvals_exc[:,1] = greens[...]
    cvals_exc[:,2] = blues[...]
    newcmp_exc = ListedColormap(cvals_exc)

    # Raster downsampling
    N_scaling = 100
    N_gap = 1

    # Firing rates plotting gap
    rates_gap = 150 # Hz

    # Power spectra parameters
    fs2 = int(1/winstep_FR)
    window_size = 100*ms
    window_width = int(fs2*window_size) # samples
    overlap = 0.99 # percentage
    noverlap = int(overlap*window_width)

    # Set theta rhythm limits
    # xlims_rhythm = [t for t in t_lims]
    xlims_rhythm = [t for t in t_lims]
    ylims_rhythm = [-0.1, 1.4]

    # Set raster limits
    xlims_raster = [t for t in t_lims]
    ylims_raster = [0, 2*N_scaling]

    # Set firing rate limits
    xlims_rates = [t for t in t_lims]
    ylims_rates = [-1, 250]

    # Set spectrogram limits
    xlims_freq = [t for t in t_lims]
    # xlims_freq[0] += window_size/2
    # xlims_freq[1] += window_size/2
    ylims_freq = [0, 150] # Hz
    zlims_freq = [0.1, 10] # cmap limits [vmin vmax]


    """ Plot Figure 2 of the paper """
    print('[+] Generating the figure...')

    # Make a figure
    fig = plt.figure(figsize=(8,8))

    # Use gridspecs
    G_outer = GridSpec(4, 2, left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.2, height_ratios=(0.15, 0.4, 0.2, 0.25), width_ratios=(0.99,0.01))
    G_rhythm = GridSpecFromSubplotSpec(1, 1, hspace=0., subplot_spec=G_outer[0,0])
    G_rasters = GridSpecFromSubplotSpec(4, 1, hspace=0.4, subplot_spec=G_outer[1,0])
    G_rates = GridSpecFromSubplotSpec(1, 1, hspace=0.6, subplot_spec=G_outer[2,0])
    G_specg = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_outer[3,0])
    G_specg_cbars = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_outer[3,1])

    G_outer.tight_layout(fig)


    # Organize axes
    #------------------------
    axs = []

    # Rasters
    # ------------------------
    print('[>] Rasters')
    ax0 = fig.add_subplot(G_rasters[0]) # EC E/I rasters
    ax1 = fig.add_subplot(G_rasters[1], sharex=ax0, sharey=ax0) # DG E/I rasters
    ax2 = fig.add_subplot(G_rasters[2], sharex=ax0, sharey=ax0) # CA3 E/I rasters
    ax3 = fig.add_subplot(G_rasters[3], sharex=ax0, sharey=ax0) # CA1 E/I rasters
    axs.append([ax0, ax1, ax2, ax3])

    # set label as area name
    # ax0.set_title('Rasters')
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


    # Firing Rates
    # ------------------------
    # ax_rate_inh = fig.add_subplot(G_rates[0], sharex=ax0)
    # ax_rate_exc = fig.add_subplot(G_rates[1], sharex=ax0, sharey=ax_rate_inh)
    # axs.append([ax_rate_exc, ax_rate_inh])
    ax_rates = fig.add_subplot(G_rates[0], sharex=ax0)
    axs.append([ax_rates])

    # Set the title
    # ax_rate_exc.set_title('CA1 Firing Rate')

    # Set the x-y limits
    # ax_rate_exc.set_ylim(ylims_rates)
    ax_rates.set_ylim([0, 400])

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
    ax_specg_inh = fig.add_subplot(G_specg[0])
    ax_specg_exc = fig.add_subplot(G_specg[1])
    axs.append([ax_specg_exc, ax_specg_inh])

    # Set the title
    # ax_specg_exc.set_title('Spectrogram')

    # Set the x-y limits
    ax_specg_inh.set_xlim(xlims_freq)
    ax_specg_inh.set_ylim(ylims_freq)
    ax_specg_exc.set_xlim(xlims_freq)
    ax_specg_exc.set_ylim(ylims_freq)

    # Set the ticks
    specg_freq_majors = [10, 40, 100]
    ax_specg_inh.yaxis.set_major_locator(ticker.FixedLocator(specg_freq_majors))
    ax_specg_exc.yaxis.set_major_locator(ticker.FixedLocator(specg_freq_majors))

    specg_freq_majors = np.arange(0., t_lims[1]/second, .5) #[0.5, 0.6, 0.7, 1., 1.25, 1.5]
    ax_specg_exc.xaxis.set_major_locator(ticker.FixedLocator(specg_freq_majors))
    ax_specg_exc.xaxis.set_minor_locator(ticker.NullLocator())

    # Hide x axis for inh
    ax_specg_inh.xaxis.set_visible(False)

    # Set xlabel
    ax_specg_exc.set_xlabel('Time [s]')

    # Hide some spines
    ax_specg_exc.spines['top'].set_visible(False)
    # ax_specg_exc.spines['bottom'].set_visible(False)
    # ax_specg_exc.spines['left'].set_visible(False)
    ax_specg_exc.spines['right'].set_visible(False)
    ax_specg_inh.spines['top'].set_visible(False)
    # ax_specg_inh.spines['bottom'].set_visible(False)
    # ax_specg_inh.spines['left'].set_visible(False)
    ax_specg_inh.spines['right'].set_visible(False)


    # Rhythm
    # ------------------------
    print('[>] Input rhythm')
    ax_rhythm = fig.add_subplot(G_rhythm[0])
    axs.append(ax_rhythm)

    # Set the title
    # ax_rhythm.set_title('Theta Input')


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
        # load t-i arrays for this area
        print('[+] Loading the spikes for area', areas[area_idx][0].split('_')[0])
        i_exc = np.loadtxt('/home/nikos/Documents/projects/Python/memstim-hh/results/analysis/current/data/spikes/{0}_spikemon_i.txt'.format(areas[area_idx][0]))
        t_exc = np.loadtxt('/home/nikos/Documents/projects/Python/memstim-hh/results/analysis/current/data/spikes/{0}_spikemon_t.txt'.format(areas[area_idx][0]))
        i_inh = np.loadtxt('/home/nikos/Documents/projects/Python/memstim-hh/results/analysis/current/data/spikes/{0}_spikemon_i.txt'.format(areas[area_idx][1]))
        t_inh = np.loadtxt('/home/nikos/Documents/projects/Python/memstim-hh/results/analysis/current/data/spikes/{0}_spikemon_t.txt'.format(areas[area_idx][1]))

        t_exc = t_exc/1000
        t_inh = t_inh/1000

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
        # exc_mixed = np.random.permutation(np.arange(N_exc))
        # inh_mixed = np.random.permutation(np.arange(N_inh))
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
        for ii in exc_mixed:
            idx_tmp = np.where(i_exc_sub == ii)
            i_exc_sub[idx_tmp] = cnt
            cnt += 1

        # cnt = 0
        for jj in inh_mixed:
            idx_tmp = np.where(i_inh_sub == jj)
            i_inh_sub[idx_tmp] = cnt
            cnt += 1

        # plot spikes
        print('[>] Plotting spikes...')

        # select axis
        ax_curr = axs[0][area_idx]

        # inhibitory
        # ax_curr.plot(t_inh_sub, i_inh_sub, 'o', c=c_inh, markersize=.25, alpha=.75, zorder=1, rasterized=False)
        ax_curr.scatter(t_inh_sub, i_inh_sub, s=1, marker='o', c=c_inh, edgecolors=None, alpha=.25, zorder=1, rasterized=False)

        # excitatory
        # ax_curr.plot(t_exc_sub, i_exc_sub, 'o', c=c_exc, markersize=.25, alpha=.75, zorder=1, rasterized=False)
        ax_curr.scatter(t_exc_sub, i_exc_sub, s=1, marker='o', c=c_exc, edgecolors=None, alpha=.25, zorder=1, rasterized=False)

        # Calculate mean firing rates
        t_lims_adj = [2000*ms, 3000*ms]
        duration_adj = t_lims_adj[1] - t_lims_adj[0]

        # FR_inh_mean = (len(t_inh)/duration)/N_inh
        # FR_exc_mean = (len(t_exc)/duration)/N_exc
        FR_inh_mean = (sum((t_inh>=t_lims_adj[0]) & (t_inh<t_lims_adj[1]))/duration_adj)/N_inh
        FR_exc_mean = (sum((t_exc>=t_lims_adj[0]) & (t_exc<t_lims_adj[1]))/duration_adj)/N_exc

        # add it as a text
        ax_curr.text(x=duration+200*ms, y=150, s=r'$\mu_I$: {0:.1f}Hz'.format(FR_inh_mean), ha='center', color=c_inh, clip_on=False)
        ax_curr.text(x=duration+200*ms, y=50, s=r'$\mu_E$: {0:.1f}Hz'.format(FR_exc_mean), ha='center', color=c_exc, clip_on=False)

    # # add a sizebar for the y-axis
    # add_sizebar(axs[0][3], [t_lims[1]+20*ms, t_lims[1]+20*ms], [0, 100], 'black', '100pts')


    # ==================
    # Plot the input rhythm
    # ==================
    print('[+] Plotting rhythm...')

    rhythm = np.loadtxt('/home/nikos/Documents/projects/Python/memstim-hh/results/analysis/current/data/order_param_mon_rhythm.txt')
    ax_rhythm.plot(np.arange(0.,duration,dt), rhythm/(np.max(rhythm)), ls='-', c='k', linewidth=1.2, rasterized=True, zorder=1)

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
    ax_rhythm.vlines(x=t_stim, ymin=-10.2, ymax=1., color='red', ls='-', linewidth=0.5, zorder=11, clip_on=False)

    # text frequency label
    ax_rhythm.text(x=duration+100*ms, y=1.1, s=r"$f_\theta={0:.2f}$Hz".format(fval), ha='left', color='k', clip_on=False)

    # add a sizebar for the y-axis
    # add_sizebar(ax_rhythm, [duration+100*ms, duration+100*ms], [0, 0.5], 'black', '0.5nA')


    # ==================
    # Plot CA1 FRs
    # ==================
    tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    FR_inh_norm = (FR_inh/winsize_FR)/N_inh
    FR_exc_norm = (FR_exc/winsize_FR)/N_exc

    # ax_rate_inh.plot(tv_inh_FR, FR_inh_norm, ls='-', linewidth=1., c=c_inh, label='inh', zorder=10, rasterized=True)
    # ax_rate_exc.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1., c=c_exc, label='exc', zorder=10, rasterized=True)
    ax_rates.plot(tv_inh_FR, FR_inh_norm+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=True)
    ax_rates.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=True)

    # add labels
    # ax_rate_inh.set_title('Inhibitory', color=c_inh, loc='left')
    # ax_rate_exc.set_title('Excitatory', color=c_exc, loc='left')

    # ax_rate_inh.text(x=-10*ms, y=ylims_rates[1]//2, s='Inhibitory', ha='center', color=c_inh, clip_on=False)
    # ax_rate_exc.text(x=-10*ms, y=ylims_rates[1]//2, s='Excitatory', ha='center', color=c_exc, clip_on=False)
    ax_rates.text(x=-100*ms, y=ylims_rates[1]-50, s='Inhibitory', ha='center', color=c_inh, clip_on=False)
    ax_rates.text(x=-100*ms, y=ylims_rates[0]+50, s='Excitatory', ha='center', color=c_exc, clip_on=False)

    # add a sizebar for the y-axis
    # add_sizebar(ax_rate_exc, [duration+100*ms, duration+100*ms], [0, 50], 'black', '50Hz')
    add_sizebar(ax_rates, [duration+100*ms, duration+100*ms], [0, 50], 'black', '50Hz')


    # ==================
    # Plot the spectrograms
    # ==================
    fv_inh, tv_inh, pspec_inh = my_specgram(signal=FR_inh_norm,
                                fs=fs2,
                                window_width=window_width,
                                window_overlap=noverlap)

    fv_exc, tv_exc, pspec_exc = my_specgram(signal=FR_exc_norm,
                                fs=fs2,
                                window_width=window_width,
                                window_overlap=noverlap)

    # avoid division by zero in log transform
    pspec_inh[np.where(pspec_inh<1e-10)] = 1e-10
    pspec_exc[np.where(pspec_exc<1e-10)] = 1e-10

    # get log power
    # pspec_inh_dB = 10*np.log10(pspec_inh)
    # pspec_exc_dB = 10*np.log10(pspec_exc)

    # find min/max for plotting
    # maxval_dB = max(pspec_inh_dB.max(), pspec_exc_dB.max())
    # minval_dB = min(pspec_inh_dB.min(), pspec_exc_dB.min())

    # pcm_inh = ax_specg_inh.pcolormesh(tv_inh, fv_inh, pspec_inh_dB, vmin=minval_dB, vmax=maxval_dB, cmap=newcmp_inh, shading='gouraud')
    # pcm_exc = ax_specg_exc.pcolormesh(tv_exc, fv_exc, pspec_exc_dB, vmin=minval_dB, vmax=maxval_dB, cmap=newcmp_exc, shading='gouraud')

    im_inh = ax_specg_inh.pcolormesh(tv_inh, fv_inh, pspec_inh, cmap=newcmp_inh, shading='auto')
    im_exc = ax_specg_exc.pcolormesh(tv_exc, fv_exc, pspec_exc, cmap=newcmp_exc, shading='auto')
    # im_inh = ax_specg_inh.imshow(pspec_inh_dB, cmap=newcmp_inh, interpolation='nearest', vmin=minval_dB, vmax=maxval_dB, origin='lower', aspect='auto')
    # im_exc = ax_specg_exc.imshow(pspec_exc_dB, cmap=newcmp_exc, interpolation='nearest', vmin=minval_dB, vmax=maxval_dB, origin='lower', aspect='auto')

    # pcm_inh= ax_specg_inh.specgram(FR_exc_norm, NFFT=window_width, detrend='none', Fs=fs2, window=sig.windows.hann(M=window_width, sym=False), noverlap=noverlap, scale_by_freq=False, mode='magnitude', scale='dB', sides='onesided')

    # Colorbars
    # fig.colorbar(pcm_exc, ax=ax_specg_exc)
    # fig.colorbar(pcm_inh, ax=ax_specg_inh)
    # cbar_inh_ax = fig.add_subplot(G_specg_cbars[0])
    # cbar_exc_ax = fig.add_subplot(G_specg_cbars[1])
    # fig.colorbar(im_inh, cax=cbar_inh_ax, aspect=1)
    # fig.colorbar(im_exc, cax=cbar_exc_ax, aspect=1)

    # sizebars
    # ax_specg_exc.plot([550*ms, 600*ms, None, 550*ms, 550*ms], [-60, -60, None, 0, 0], ls='-', c='r', linewidth=1., rasterized=True, clip_on=False)
    # ax_specg_exc.text(x=575*ms, y=-100, s='50ms', ha='center', fontproperties=fontprops, clip_on=False)

    # save the figure
    print('[+] Saving the figures...')
    # fig.savefig('figures/fig2_A.svg', dpi=200, format='svg')
    # fig.savefig('figures/fig2_A.pdf', dpi=200, format='pdf')
    fig.savefig('figures/fig2_A.png', dpi=200, format='png')


    # Also make an animation
    # ------------------------
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
    # print('[+] Making animation...')
    # line_animation = animation.FuncAnimation(fig, update_line, frames=F, interval=1e3/framerate, blit=True)

    # print('[+] Saving the video...')
    # line_animation.save('/home/nikos/Documents/projects/Python/memstim-hh/figures/test.mp4', fps=framerate, extra_args=['-vcodec', 'libx264'])
    # save animation at 30 frames per second
    # line_animation.save('/home/nikos/Documents/projects/Python/memstim-hh/figures/test.gif', writer='imagemagick', fps=framerate)
    print('[!] Done')

    # fig.tight_layout()
    plt.show()

    exit(0)
