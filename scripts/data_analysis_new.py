#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import numpy as np
# import matplotlib.pyplot as plt

# from brian2 import *

import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

from scipy import signal as sig


fontprops = fm.FontProperties(size=12, family='monospace')

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42

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
    return np.array(counts), centers


def my_FR_hist(spikes: np.ndarray,
            duration: int,
            window_size: float) -> (np.ndarray, np.ndarray):
    """
    Compute the firing rate using numpy's histogram function.

    Parameters
    ----------
    spikes: numpy.ndarray
        The spike times (Brian2 format, in ms)
    duration: int
        The duration of the recording (in seconds)
    window_size: float
        Width of the window to calculate spikes in (in seconds)

    Returns
    -------
    t: numpy.ndarray
        Array of time values for the computed firing rate. These are the window centers.
    FR: numpy.ndarray
        Spikes per window (needs to be normalized)
    """

    bin_num = int(duration/window_size)

    spk_cnt, bin_edges = np.histogram(spikes, bins=bin_num, range=(0,duration))

    return spk_cnt, bin_edges[:-1] + window_size/2


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
    k = 3
    nfft = 2**k * window_width # np.ceil(np.log2(window_width))
    f, t, Sxx = sig.spectrogram(x=signal,
                                nfft=nfft,
                                detrend=False,
                                fs=fs,
                                window=sig.windows.hann(M=window_width, sym=False),
                                # nperseg=window_width,
                                noverlap=window_overlap,
                                return_onesided=True,
                                scaling='spectrum',
                                mode='magnitude')

    return f, t, Sxx
    # ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel


def calc_PLV(sigl:np.ndarray,
             sigh:np.ndarray) -> (np.ndarray, tuple):
    '''
    Compute the Phase-Locking Value (PLV) as mentioned in [1]-[2]

    Args:
    ============
    sigl: numpy.ndarray
        Lower-frequency signal
    sigh: numpy.ndarray
        Higher-frequency signal

    Returns:
    ============
    lags : numpy.ndarray
        Phase lag values between the two signals at each data point
    PLV : tuple
        PLV[0]: Mean phase lag amplitude (actual PLV)
        PLV[1]: Mean phase lag angle
    '''

    # 1. Hilbert-transform lower-frequency signal
    sigl_a = sig.hilbert(sigl)

    # 2. Hilbert-transform higher-frequency signal
    sigh_a = sig.hilbert(sigh)

    # 3. Hilbert-transform the amplitude of the HF-analytic signal
    sigh_aa = sig.hilbert(np.real(sigh_a))

    # 4. Calculate the phase differences
    phi_lt = np.angle(sigl_a)
    phi_ht = np.angle(sigh_aa)
    phi_d = phi_lt - phi_ht
    lags = np.exp(1j*phi_d)

    # 5. Calculate the PLV (mean of lags)
    mu_phi = np.mean(lags)
    PLV = (np.abs(mu_phi), np.angle(mu_phi))

    return lags, PLV


def myround(x, base=100):
    """ Rounding function to closest *base* """
    val = int(base * round(x/base))
    return val if val > 0 else base*1



# main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Data analysis script')

    parser.add_argument('-m', '--method',
                        type=str,
                        default='histogram',
                        help='Which method to use for the calculation of the FRs. [histogram/hist] | [windowed/win]')

    args = parser.parse_args()


    # Parameters
    # ------------------------------------------------------------------------------
    print('[+] Setting up parameters...')

    # Timings
    second = sec = 1.
    millisecond = ms = 1e-3

    # Populations
    areas = ['EC', 'DG', 'CA3', 'CA1']
    EC_exc_N = 10000
    EC_inh_N = 1000
    DG_exc_N = 10000
    DG_inh_N = 100
    CA3_exc_N = 1000
    CA3_inh_N = 100
    CA1_exc_N = 10000
    CA1_inh_N = 1000
    N_tot = EC_exc_N + EC_inh_N + DG_exc_N + DG_inh_N + CA3_exc_N + CA3_inh_N + CA1_exc_N + CA1_inh_N

    # Timing
    duration = 2*second
    dt = 0.1*ms
    fs = int(1/dt)

    # FR calculation
    winsize_FR = 1*ms
    overlap_FR = 0.9
    winstep_FR = winsize_FR*round(1-overlap_FR,4)

    # Choice of FR calculation method
    if args.method.lower() == "histogram" or args.method.lower() == "hist":
        # use histogram method
        print("[!] Using histogram method for firing rate calculation")
        calc_FR = lambda spikes: my_FR_hist(spikes=spikes, duration=duration, window_size=winsize_FR)
        fs_FR = int(1/winsize_FR)

    else:
        # use my own method (same as fig2_A calculations)
        print("[!] Using custom method for firing rate calculation")
        calc_FR = lambda spikes: my_FR(spikes=spikes, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
        fs_FR = int(1/winstep_FR)

    # PSD calculation
    freq_res = 2. # 1 Hz freq. res -> 1 second window
    winsize_freq = 1./freq_res
    winsize_samples = winsize_freq*fs_FR

    # Time-frequency parameters
    winsize_TF = 100*ms
    winsize_samples_TF = int(fs_FR*winsize_TF) # samples
    overlap_TF = 0.9 # percentage
    noverlap_TF = int(overlap_TF*winsize_samples_TF)


    # Directories
    # ------------------------------------------------------------------------------
    dirs = {}

    # Results path
    dirs['results'] = 'results/analysis/current/desc/'
    dirs['data'] = dirs['results'] + 'data/'

    # Spikes
    dirs['spikes'] = dirs['data'] + 'spikes/'


    # Load the rasters
    # ------------------------------------------------------------------------------
    print("[+] Loading rasters...")

    EC_exc_t = np.loadtxt(dirs['spikes'] + 'EC_pyCAN_spikemon_t.txt', dtype=np.float32)
    EC_exc_i = np.loadtxt(dirs['spikes'] + 'EC_pyCAN_spikemon_i.txt', dtype=int)
    EC_inh_t = np.loadtxt(dirs['spikes'] + 'EC_inh_spikemon_t.txt', dtype=np.float32)
    EC_inh_i = np.loadtxt(dirs['spikes'] + 'EC_inh_spikemon_i.txt', dtype=int)

    DG_exc_t = np.loadtxt(dirs['spikes'] + 'DG_py_spikemon_t.txt', dtype=np.float32)
    DG_exc_i = np.loadtxt(dirs['spikes'] + 'DG_py_spikemon_i.txt', dtype=int)
    DG_inh_t = np.loadtxt(dirs['spikes'] + 'DG_inh_spikemon_t.txt', dtype=np.float32)
    DG_inh_i = np.loadtxt(dirs['spikes'] + 'DG_inh_spikemon_i.txt', dtype=int)

    CA3_exc_t = np.loadtxt(dirs['spikes'] + 'CA3_pyCAN_spikemon_t.txt', dtype=np.float32)
    CA3_exc_i = np.loadtxt(dirs['spikes'] + 'CA3_pyCAN_spikemon_i.txt', dtype=int)
    CA3_inh_t = np.loadtxt(dirs['spikes'] + 'CA3_inh_spikemon_t.txt', dtype=np.float32)
    CA3_inh_i = np.loadtxt(dirs['spikes'] + 'CA3_inh_spikemon_i.txt', dtype=int)

    CA1_exc_t = np.loadtxt(dirs['spikes'] + 'CA1_pyCAN_spikemon_t.txt', dtype=np.float32)
    CA1_exc_i = np.loadtxt(dirs['spikes'] + 'CA1_pyCAN_spikemon_i.txt', dtype=int)
    CA1_inh_t = np.loadtxt(dirs['spikes'] + 'CA1_inh_spikemon_t.txt', dtype=np.float32)
    CA1_inh_i = np.loadtxt(dirs['spikes'] + 'CA1_inh_spikemon_i.txt', dtype=int)


    # Generate firing rates (LFP estimates)
    # COBA example, brian2 docs
    # https://brian2.readthedocs.io/en/stable/examples/frompapers.Stimberg_et_al_2018.example_1_COBA.html
    # ------------------------------------------------------------------------------
    print("[+] Generating firing rates from rasters...")
    EC_exc_spk_cnt, EC_exc_bin_edges = calc_FR(EC_exc_t*ms)
    EC_exc_rate = (EC_exc_spk_cnt/winsize_FR)/(EC_exc_N)
    EC_inh_spk_cnt, EC_inh_bin_edges = calc_FR(EC_inh_t*ms)
    EC_inh_rate = (EC_inh_spk_cnt/winsize_FR)/(EC_inh_N)

    DG_exc_spk_cnt, DG_exc_bin_edges = calc_FR(DG_exc_t*ms)
    DG_exc_rate = (DG_exc_spk_cnt/winsize_FR)/(DG_exc_N)
    DG_inh_spk_cnt, DG_inh_bin_edges = calc_FR(DG_inh_t*ms)
    DG_inh_rate = (DG_inh_spk_cnt/winsize_FR)/(DG_inh_N)

    CA3_exc_spk_cnt, CA3_exc_bin_edges = calc_FR(CA3_exc_t*ms)
    CA3_exc_rate = (CA3_exc_spk_cnt/winsize_FR)/(CA3_exc_N)
    CA3_inh_spk_cnt, CA3_inh_bin_edges = calc_FR(CA3_inh_t*ms)
    CA3_inh_rate = (CA3_inh_spk_cnt/winsize_FR)/(CA3_inh_N)

    CA1_exc_spk_cnt, CA1_exc_bin_edges = calc_FR(CA1_exc_t*ms)
    CA1_exc_rate = (CA1_exc_spk_cnt/winsize_FR)/(CA1_exc_N)
    CA1_inh_spk_cnt, CA1_inh_bin_edges = calc_FR(CA1_inh_t*ms)
    CA1_inh_rate = (CA1_inh_spk_cnt/winsize_FR)/(CA1_inh_N)


    # Plot the firing rates
    # ------------------------------------------------------------------------------
    print("[+] Plotting firing rates...")
    fig1, axs1 = plt.subplots(4, sharex=True, sharey=True, figsize=(16,8))
    axs1[0].plot(EC_exc_bin_edges, EC_exc_rate, linestyle='-', color='b', label= 'exc')
    axs1[0].plot(EC_inh_bin_edges, EC_inh_rate, linestyle='-', color='r', label='inh')
    axs1[1].plot(DG_exc_bin_edges, DG_exc_rate, linestyle='-', color='b', label='exc')
    axs1[1].plot(DG_inh_bin_edges, DG_inh_rate, linestyle='-', color='r', label='inh')
    axs1[2].plot(CA3_exc_bin_edges, CA3_exc_rate, linestyle='-', color='b', label='exc')
    axs1[2].plot(CA3_inh_bin_edges, CA3_inh_rate, linestyle='-', color='r', label='inh')
    axs1[3].plot(CA1_exc_bin_edges, CA1_exc_rate, linestyle='-', color='b', label='exc')
    axs1[3].plot(CA1_inh_bin_edges, CA1_inh_rate, linestyle='-', color='r', label='inh')

    for (ax, lbl) in zip(axs1, areas):
        ax.set_title(lbl)
        ax.set_ylabel('Rate [Hz]', rotation=0, fontsize=12, labelpad=30)
        ax.grid(linestyle='-', color=[0.5, 0.5, 0.5], alpha=0.33)
        ax.set_ylim([0, 1000])

        axs1[3].set_xlabel('Time [ms]')
        axs1[3].legend()


    # Analysis #1: Frequency (PSD)
    # ------------------------------------------------------------------------------
    from scipy.fft import rfft, rfftfreq

    print("\n[*] Analysis #1")
    print("[+] Calculating PSDs...")

    # win = 'boxcar' # periodogram default
    win = 'hann'
    N = len(EC_exc_rate)
    NFFT = 2*int(pow(2, np.ceil(np.log(2*N)/np.log(2))))
    dt2 = winsize_FR
    fs2 = 1/dt2

    # fv = rfftfreq(NFFT, dt2)
    # EC_exc_Pxx = np.abs(rfft(EC_exc_rate, n=NFFT)/NFFT)**2
    # EC_inh_Pxx = np.abs(rfft(EC_inh_rate, n=NFFT)/NFFT)**2
    # DG_exc_Pxx = np.abs(rfft(DG_exc_rate, n=NFFT)/NFFT)**2
    # DG_inh_Pxx = np.abs(rfft(DG_inh_rate, n=NFFT)/NFFT)**2
    # CA3_exc_Pxx = np.abs(rfft(CA3_exc_rate, n=NFFT)/NFFT)**2
    # CA3_inh_Pxx = np.abs(rfft(CA3_inh_rate, n=NFFT)/NFFT)**2
    # CA1_exc_Pxx = np.abs(rfft(CA1_exc_rate, n=NFFT)/NFFT)**2
    # CA1_inh_Pxx = np.abs(rfft(CA1_inh_rate, n=NFFT)/NFFT)**2

    fv, EC_exc_Pxx = sig.welch(EC_exc_rate, fs=fs_FR, window=win, nperseg=winsize_samples, nfft=NFFT, scaling='spectrum', average='mean')
    _, EC_inh_Pxx = sig.welch(EC_inh_rate, fs=fs_FR, window=win, nperseg=winsize_samples, nfft=NFFT, scaling='spectrum', average='mean')
    _, DG_exc_Pxx = sig.welch(DG_exc_rate, fs=fs_FR, window=win, nperseg=winsize_samples, nfft=NFFT, scaling='spectrum', average='mean')
    _, DG_inh_Pxx = sig.welch(DG_inh_rate, fs=fs_FR, window=win, nperseg=winsize_samples, nfft=NFFT, scaling='spectrum', average='mean')
    _, CA3_exc_Pxx = sig.welch(CA3_exc_rate, fs=fs_FR, window=win, nperseg=winsize_samples, nfft=NFFT, scaling='spectrum', average='mean')
    _, CA3_inh_Pxx = sig.welch(CA3_inh_rate, fs=fs_FR, window=win, nperseg=winsize_samples, nfft=NFFT, scaling='spectrum', average='mean')
    _, CA1_exc_Pxx = sig.welch(CA1_exc_rate, fs=fs_FR, window=win, nperseg=winsize_samples, nfft=NFFT, scaling='spectrum', average='mean')
    _, CA1_inh_Pxx = sig.welch(CA1_inh_rate, fs=fs_FR, window=win, nperseg=winsize_samples, nfft=NFFT, scaling='spectrum', average='mean')

    # Plot the PSDs
    # ------------------------------------------------------------------------------
    fig2, axs2 = plt.subplots(4, sharex=True, sharey=True, figsize=(16,8))
    axs2[0].plot(fv, EC_exc_Pxx, linestyle='-', color='b', label='exc')
    axs2[0].plot(fv, EC_inh_Pxx, linestyle='-', color='r', label='inh')
    axs2[1].plot(fv, DG_exc_Pxx, linestyle='-', color='b', label='exc')
    axs2[1].plot(fv, DG_inh_Pxx, linestyle='-', color='r', label='inh')
    axs2[2].plot(fv, CA3_exc_Pxx, linestyle='-', color='b', label='exc')
    axs2[2].plot(fv, CA3_inh_Pxx, linestyle='-', color='r', label='inh')
    axs2[3].plot(fv, CA1_exc_Pxx, linestyle='-', color='b', label='exc')
    axs2[3].plot(fv, CA1_inh_Pxx, linestyle='-', color='r', label='inh')

    for (ax, lbl) in zip(axs2, areas):
        ax.set_title(lbl)
        # ax.set_ylabel('Rate [Hz]', rotation=0, fontsize=12, labelpad=30)
        ax.grid(linestyle='-', color=[0.5, 0.5, 0.5], alpha=0.33)

    axs2[3].set_xlim([0,150])
    axs2[3].set_xlabel('Time [ms]')
    axs2[3].legend()


    # Analysis #2a: Time-Frequency % TODO: Fix Z-limits and add colorbar
    # ------------------------------------------------------------------------------
    print("\n[*] Analysis #2a")
    print("[+] Time-frequency analysis (scipy spectrogram)...")

    # win_size_t = 0.25
    # win_size_s = int(win_size_t * fs2)

    # fv, tv, EC_exc_Sxx = sig.spectrogram(EC_exc_rate, fs=int(fs2), nfft=NFFT, window=sig.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))
    # EC_inh_Sxx = sig.spectrogram(EC_inh_rate, fs=int(fs2), nfft=NFFT, window=sig.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
    # DG_exc_Sxx = sig.spectrogram(DG_exc_rate, fs=int(fs2), nfft=NFFT, window=sig.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
    # DG_inh_Sxx = sig.spectrogram(DG_inh_rate, fs=int(fs2), nfft=NFFT, window=sig.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
    # CA3_exc_Sxx = sig.spectrogram(CA3_exc_rate, fs=int(fs2), nfft=NFFT, window=sig.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
    # CA3_inh_Sxx = sig.spectrogram(CA3_inh_rate, fs=int(fs2), nfft=NFFT, window=sig.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
    # CA1_exc_Sxx = sig.spectrogram(CA1_exc_rate, fs=int(fs2), nfft=NFFT, window=sig.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
    # CA1_inh_Sxx = sig.spectrogram(CA1_inh_rate, fs=int(fs2), nfft=NFFT, window=sig.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]

    # fig3, axs3 = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(16,8))
    # axs3[0,0].pcolormesh(tv, fv, 1./win_size_s*EC_exc_Sxx**2, cmap='inferno', shading='gouraud')
    # axs3[1,0].pcolormesh(tv, fv, 1./win_size_s*EC_inh_Sxx**2, cmap='inferno', shading='gouraud')
    # axs3[0,1].pcolormesh(tv, fv, 1./win_size_s*DG_exc_Sxx**2, cmap='inferno', shading='gouraud')
    # axs3[1,1].pcolormesh(tv, fv, 1./win_size_s*DG_inh_Sxx**2, cmap='inferno', shading='gouraud')
    # axs3[0,2].pcolormesh(tv, fv, 1./win_size_s*CA3_exc_Sxx**2, cmap='inferno', shading='gouraud')
    # axs3[1,2].pcolormesh(tv, fv, 1./win_size_s*CA3_inh_Sxx**2, cmap='inferno', shading='gouraud')
    # axs3[0,3].pcolormesh(tv, fv, 1./win_size_s*CA1_exc_Sxx**2, cmap='inferno', shading='gouraud')
    # axs3[1,3].pcolormesh(tv, fv, 1./win_size_s*CA1_inh_Sxx**2, cmap='inferno', shading='gouraud')

    fv, tv, EC_exc_Sxx = my_specgram(signal=EC_exc_rate, fs=fs_FR, window_width=winsize_samples_TF, window_overlap=noverlap_TF)
    _, _, EC_inh_Sxx = my_specgram(signal=EC_inh_rate, fs=fs_FR, window_width=winsize_samples_TF, window_overlap=noverlap_TF)
    _, _, DG_exc_Sxx = my_specgram(signal=DG_exc_rate, fs=fs_FR, window_width=winsize_samples_TF, window_overlap=noverlap_TF)
    _, _, DG_inh_Sxx = my_specgram(signal=DG_inh_rate, fs=fs_FR, window_width=winsize_samples_TF, window_overlap=noverlap_TF)
    _, _, CA3_exc_Sxx = my_specgram(signal=CA3_exc_rate, fs=fs_FR, window_width=winsize_samples_TF, window_overlap=noverlap_TF)
    _, _, CA3_inh_Sxx = my_specgram(signal=CA3_inh_rate, fs=fs_FR, window_width=winsize_samples_TF, window_overlap=noverlap_TF)
    _, _, CA1_exc_Sxx = my_specgram(signal=CA1_exc_rate, fs=fs_FR, window_width=winsize_samples_TF, window_overlap=noverlap_TF)
    _, _, CA1_inh_Sxx = my_specgram(signal=CA1_inh_rate, fs=fs_FR, window_width=winsize_samples_TF, window_overlap=noverlap_TF)

    fig3, axs3 = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(16,8))
    axs3[0,0].pcolormesh(tv, fv, EC_exc_Sxx/EC_exc_Sxx.max(), cmap='inferno', shading='auto', rasterized=True)
    axs3[1,0].pcolormesh(tv, fv, EC_inh_Sxx/EC_inh_Sxx.max(), cmap='inferno', shading='auto', rasterized=True)
    axs3[0,1].pcolormesh(tv, fv, DG_exc_Sxx/DG_exc_Sxx.max(), cmap='inferno', shading='auto', rasterized=True)
    axs3[1,1].pcolormesh(tv, fv, DG_inh_Sxx/DG_inh_Sxx.max(), cmap='inferno', shading='auto', rasterized=True)
    axs3[0,2].pcolormesh(tv, fv, CA3_exc_Sxx/CA3_exc_Sxx.max(), cmap='inferno', shading='auto', rasterized=True)
    axs3[1,2].pcolormesh(tv, fv, CA3_inh_Sxx/CA3_inh_Sxx.max(), cmap='inferno', shading='auto', rasterized=True)
    axs3[0,3].pcolormesh(tv, fv, CA1_exc_Sxx/CA1_exc_Sxx.max(), cmap='inferno', shading='auto', rasterized=True)
    axs3[1,3].pcolormesh(tv, fv, CA1_inh_Sxx/CA1_inh_Sxx.max(), cmap='inferno', shading='auto', rasterized=True)

    for ax_out in axs3:
        for ax_in in ax_out:
            ax_in.set_ylabel('Frequency [Hz]')
            ax_in.set_xlabel('Time [sec]')
            ax_in.set_ylim(0,150)

    axs3[0,0].set_title('EC')
    axs3[0,1].set_title('DG')
    axs3[0,2].set_title('CA3')
    axs3[0,3].set_title('CA1')

    fig3.suptitle('Spectrograms', fontsize=16)


    # Analysis #2b: Time-Frequency (wavelets?)
    # ------------------------------------------------------------------------------
    print("\n[*] Analysis #2b")
    print("[+] Time-frequency analysis (spectrogram) using wavelets...")
    print(":: PASS ::")


    # Analysis #3: PAC of theta-gamma rhythms
    # https://www.frontiersin.org/articles/10.3389/fnins.2019.00573/full#:~:text=To%20calculate%20phase%2Damplitude%20coupling%2C%20first%2C%20raw%20data%20is,the%20complex%2Dvalued%20analytic%20signal.
    # ------------------------------------------------------------------------------
    print("\n[*] Analysis #3")
    print("[+] Phase-amplitude coupling in theta-gamma using filter-Hilbert method...")

    def butter_lowpass(lowcut, fs, order=8, sos=False):
        ''' Create a lowpass butterworth filter '''
        nyq = 0.5 * fs
        low = lowcut / nyq

        if sos:
            sos_out = sig.butter(order, low, analog=False, btype='low', output='sos')
            return sos_out

        b, a = sig.butter(order, low, analog=False, btype='low', output='ba')
        return b, a

    def butter_highpass(highcut, fs, order=8, sos=False):
        ''' Create a highpass butterworth filter '''
        nyq = 0.5 * fs
        high = highcut / nyq

        if sos:
            sos_out = sig.butter(order, high, analog=False, btype='high', output='sos')
            return sos_out

        b, a = sig.butter(order, high, analog=False, btype='high', output='ba')
        return b, a

    def butter_bandpass(lowcut, highcut, fs, order=8, sos=False):
        ''' Create a bandpass butterworth filter '''
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        if sos:
            sos_out = sig.butter(order, [low, high], analog=False, btype='band', output='sos')
            return sos_out

        b, a = sig.butter(order, [low, high], analog=False, btype='band', output='ba')
        return b, a


    def butter_lowpass_filter(data, lowcut, fs, order=5, sos=False):
        ''' Lowpass filter the data '''
        if sos:
            sos_out = butter_lowpass(lowcut, fs, order=order, sos=sos)
            y = sig.sosfiltfilt(sos_out, data)
        else:
            b, a = butter_lowpass(lowcut, fs, order=order, sos=sos)
            y = sig.filtfilt(b, a, data)

        return y

    def butter_highpass_filter(data, highcut, fs, order=5, sos=False):
        ''' Highpass filter the data '''
        if sos:
            sos_out = butter_highpass(highcut, fs, order=order, sos=sos)
            y = sig.sosfiltfilt(sos_out, data)
        else:
            b, a = butter_highpass(highcut, fs, order=order, sos=sos)
            y = sig.filtfilt(b, a, data)

        return y

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, sos=False):
        ''' Bandpass filter the data '''
        if sos:
            sos_out = butter_bandpass(lowcut, highcut, fs, order=order, sos=sos)
            y = sig.sosfiltfilt(sos_out, data)
        else:
            b, a = butter_bandpass(lowcut, highcut, fs, order=order, sos=sos)
            y = sig.filtfilt(b, a, data)

        return y


    # Make butterworth n-th order bandpass filters for theta/gamma bands
    # Theta band filter
    N = 12 # filter order
    fc_theta = [1, 10] # theta frequencies [Hz]
    fc_gamma = [30, 200] # all gamma frequencyes [Hz]
    fc_gamma_low = [30, 60] # low-gamma frequencies [Hz]
    fc_gamma_high = [60, 120] # high-gamma frequencies [Hz]

    # Theta band filter
    b, a = butter_bandpass(fc_theta[0], fc_theta[1], fs=int(fs_FR), order=N, sos=False)
    sos = butter_bandpass(fc_theta[0], fc_theta[1], fs=int(fs_FR), order=N, sos=True)
    filt_theta = {'a':a, 'b':b, 'sos':sos}

    # Gamma band filter
    b, a = butter_bandpass(fc_gamma[0], fc_gamma[1], fs=int(fs_FR), order=N, sos=False)
    sos = butter_bandpass(fc_gamma[0], fc_gamma[1], fs=int(fs_FR), order=N, sos=True)
    filt_gamma = {'a':a, 'b':b, 'sos':sos}

    # Low-gamma filter
    b, a = butter_bandpass(fc_gamma_low[0], fc_gamma_low[1], fs=int(fs_FR), order=N, sos=False)
    sos = butter_bandpass(fc_gamma_low[0], fc_gamma_low[1], fs=int(fs_FR), order=N, sos=True)
    filt_gamma_low = {'a':a,'b':b, 'sos':sos}

    # High-gamma filter
    b, a = butter_bandpass(fc_gamma_high[0], fc_gamma_high[1], fs=int(fs_FR), order=N, sos=False)
    sos = butter_bandpass(fc_gamma_high[0], fc_gamma_high[1], fs=int(fs_FR), order=N, sos=True)
    filt_gamma_high = {'a':a,'b':b, 'sos':sos}

    # Filtering
    data = EC_exc_rate
    EC_exc_rate_filt_low = butter_lowpass_filter(data, lowcut=12, fs=int(fs_FR), order=N, sos=True)
    EC_exc_rate_filt_theta = sig.sosfiltfilt(sos=filt_theta['sos'], x=data)
    EC_exc_rate_filt_gamma = sig.sosfiltfilt(sos=filt_gamma['sos'], x=data)
    EC_exc_rate_filt_gamma_low = sig.sosfiltfilt(sos=filt_gamma_low['sos'], x=data)
    EC_exc_rate_filt_gamma_high = sig.sosfiltfilt(sos=filt_gamma_high['sos'], x=data)
    EC_exc_rate_filt_high = butter_highpass_filter(data, highcut=30, fs=int(fs_FR), order=N, sos=True)

    data = EC_inh_rate
    EC_inh_rate_filt_low = butter_lowpass_filter(data, lowcut=12, fs=int(fs_FR), order=N, sos=True)
    EC_inh_rate_filt_theta = sig.sosfiltfilt(sos=filt_theta['sos'], x=data)
    EC_inh_rate_filt_gamma = sig.sosfiltfilt(sos=filt_gamma['sos'], x=data)
    EC_inh_rate_filt_gamma_low = sig.sosfiltfilt(sos=filt_gamma_low['sos'], x=data)
    EC_inh_rate_filt_gamma_high = sig.sosfiltfilt(sos=filt_gamma_high['sos'], x=data)
    EC_inh_rate_filt_high = butter_highpass_filter(data, highcut=30, fs=int(fs_FR), order=N, sos=True)

    data = DG_exc_rate
    DG_exc_rate_filt_low = butter_lowpass_filter(data, lowcut=12, fs=int(fs_FR), order=N, sos=True)
    DG_exc_rate_filt_theta = sig.sosfiltfilt(sos=filt_theta['sos'], x=data)
    DG_exc_rate_filt_gamma = sig.sosfiltfilt(sos=filt_gamma['sos'], x=data)
    DG_exc_rate_filt_gamma_low = sig.sosfiltfilt(sos=filt_gamma_low['sos'], x=data)
    DG_exc_rate_filt_gamma_high = sig.sosfiltfilt(sos=filt_gamma_high['sos'], x=data)
    DG_exc_rate_filt_high = butter_highpass_filter(data, highcut=30, fs=int(fs_FR), order=N, sos=True)

    data = DG_inh_rate
    DG_inh_rate_filt_low = butter_lowpass_filter(data, lowcut=12, fs=int(fs_FR), order=N, sos=True)
    DG_inh_rate_filt_theta = sig.sosfiltfilt(sos=filt_theta['sos'], x=data)
    DG_inh_rate_filt_gamma = sig.sosfiltfilt(sos=filt_gamma['sos'], x=data)
    DG_inh_rate_filt_gamma_low = sig.sosfiltfilt(sos=filt_gamma_low['sos'], x=data)
    DG_inh_rate_filt_gamma_high = sig.sosfiltfilt(sos=filt_gamma_high['sos'], x=data)
    DG_inh_rate_filt_high = butter_highpass_filter(data, highcut=30, fs=int(fs_FR), order=N, sos=True)

    data = CA3_exc_rate
    CA3_exc_rate_filt_low = butter_lowpass_filter(data, lowcut=12, fs=int(fs_FR), order=N, sos=True)
    CA3_exc_rate_filt_theta = sig.sosfiltfilt(sos=filt_theta['sos'], x=data)
    CA3_exc_rate_filt_gamma = sig.sosfiltfilt(sos=filt_gamma['sos'], x=data)
    CA3_exc_rate_filt_gamma_low = sig.sosfiltfilt(sos=filt_gamma_low['sos'], x=data)
    CA3_exc_rate_filt_gamma_high = sig.sosfiltfilt(sos=filt_gamma_high['sos'], x=data)
    CA3_exc_rate_filt_high = butter_highpass_filter(data, highcut=30, fs=int(fs_FR), order=N, sos=True)

    data = CA3_inh_rate
    CA3_inh_rate_filt_low = butter_lowpass_filter(data, lowcut=12, fs=int(fs_FR), order=N, sos=True)
    CA3_inh_rate_filt_theta = sig.sosfiltfilt(sos=filt_theta['sos'], x=data)
    CA3_inh_rate_filt_gamma = sig.sosfiltfilt(sos=filt_gamma['sos'], x=data)
    CA3_inh_rate_filt_gamma_low = sig.sosfiltfilt(sos=filt_gamma_low['sos'], x=data)
    CA3_inh_rate_filt_gamma_high = sig.sosfiltfilt(sos=filt_gamma_high['sos'], x=data)
    CA3_inh_rate_filt_high = butter_highpass_filter(data, highcut=30, fs=int(fs_FR), order=N, sos=True)

    data = CA1_exc_rate
    CA1_exc_rate_filt_low = butter_lowpass_filter(data, lowcut=12, fs=int(fs_FR), order=N, sos=True)
    CA1_exc_rate_filt_theta = sig.sosfiltfilt(sos=filt_theta['sos'], x=data)
    CA1_exc_rate_filt_gamma = sig.sosfiltfilt(sos=filt_gamma['sos'], x=data)
    CA1_exc_rate_filt_gamma_low = sig.sosfiltfilt(sos=filt_gamma_low['sos'], x=data)
    CA1_exc_rate_filt_gamma_high = sig.sosfiltfilt(sos=filt_gamma_high['sos'], x=data)
    CA1_exc_rate_filt_high = butter_highpass_filter(data, highcut=30, fs=int(fs_FR), order=N, sos=True)

    data = CA1_inh_rate
    CA1_inh_rate_filt_low = butter_lowpass_filter(data, lowcut=12, fs=int(fs_FR), order=N, sos=True)
    CA1_inh_rate_filt_theta = sig.sosfiltfilt(sos=filt_theta['sos'], x=data)
    CA1_inh_rate_filt_gamma = sig.sosfiltfilt(sos=filt_gamma['sos'], x=data)
    CA1_inh_rate_filt_gamma_low = sig.sosfiltfilt(sos=filt_gamma_low['sos'], x=data)
    CA1_inh_rate_filt_gamma_high = sig.sosfiltfilt(sos=filt_gamma_high['sos'], x=data)
    CA1_inh_rate_filt_high = butter_highpass_filter(data, highcut=30, fs=int(fs_FR), order=N, sos=True)

    # Analytic signals from Hilbert transform
    # Theta
    EC_exc_theta_a = sig.hilbert(EC_exc_rate_filt_theta)
    EC_inh_theta_a = sig.hilbert(EC_inh_rate_filt_theta)
    DG_exc_theta_a = sig.hilbert(DG_exc_rate_filt_theta)
    DG_inh_theta_a = sig.hilbert(DG_inh_rate_filt_theta)
    CA3_exc_theta_a = sig.hilbert(CA3_exc_rate_filt_theta)
    CA3_inh_theta_a = sig.hilbert(CA3_inh_rate_filt_theta)
    CA1_exc_theta_a = sig.hilbert(CA1_exc_rate_filt_theta)
    CA1_inh_theta_a = sig.hilbert(CA1_inh_rate_filt_theta)

    # Gamma
    EC_exc_gamma_a = sig.hilbert(EC_exc_rate_filt_gamma)
    EC_inh_gamma_a = sig.hilbert(EC_inh_rate_filt_gamma)
    DG_exc_gamma_a = sig.hilbert(DG_exc_rate_filt_gamma)
    DG_inh_gamma_a = sig.hilbert(DG_inh_rate_filt_gamma)
    CA3_exc_gamma_a = sig.hilbert(CA3_exc_rate_filt_gamma)
    CA3_inh_gamma_a = sig.hilbert(CA3_inh_rate_filt_gamma)
    CA1_exc_gamma_a = sig.hilbert(CA1_exc_rate_filt_gamma)
    CA1_inh_gamma_a = sig.hilbert(CA1_inh_rate_filt_gamma)

    # Gamma (low)
    EC_exc_gamma_low_a = sig.hilbert(EC_exc_rate_filt_gamma_low)
    EC_inh_gamma_low_a = sig.hilbert(EC_inh_rate_filt_gamma_low)
    DG_exc_gamma_low_a = sig.hilbert(DG_exc_rate_filt_gamma_low)
    DG_inh_gamma_low_a = sig.hilbert(DG_inh_rate_filt_gamma_low)
    CA3_exc_gamma_low_a = sig.hilbert(CA3_exc_rate_filt_gamma_low)
    CA3_inh_gamma_low_a = sig.hilbert(CA3_inh_rate_filt_gamma_low)
    CA1_exc_gamma_low_a = sig.hilbert(CA1_exc_rate_filt_gamma_low)
    CA1_inh_gamma_low_a = sig.hilbert(CA1_inh_rate_filt_gamma_low)

    # Gamma (high)
    EC_exc_gamma_high_a = sig.hilbert(EC_exc_rate_filt_gamma_high)
    EC_inh_gamma_high_a = sig.hilbert(EC_inh_rate_filt_gamma_high)
    DG_exc_gamma_high_a = sig.hilbert(DG_exc_rate_filt_gamma_high)
    DG_inh_gamma_high_a = sig.hilbert(DG_inh_rate_filt_gamma_high)
    CA3_exc_gamma_high_a = sig.hilbert(CA3_exc_rate_filt_gamma_high)
    CA3_inh_gamma_high_a = sig.hilbert(CA3_inh_rate_filt_gamma_high)
    CA1_exc_gamma_high_a = sig.hilbert(CA1_exc_rate_filt_gamma_high)
    CA1_inh_gamma_high_a = sig.hilbert(CA1_inh_rate_filt_gamma_high)


    # Plot the filtered rates -- TODO: Make axes pretty (legends, labels, grids, titles etc)
    # ------------------------------------------------------------------------------
    print("[+] Plotting filtered rates...")
    fig4, axs4 = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(16,8))

    axs4[0,0].plot(EC_exc_bin_edges, EC_exc_rate, linestyle='-', color='b', label= 'exc')
    axs4[0,0].plot(EC_exc_bin_edges, EC_exc_rate_filt_theta, linestyle='-', color='C1', label= 'Theta')
    axs4[0,0].plot(EC_exc_bin_edges, EC_exc_rate_filt_gamma, linestyle='-', color='C2', label= 'Gamma')
    axs4[0,1].plot(EC_inh_bin_edges, EC_inh_rate, linestyle='-', color='r', label= 'inh')
    axs4[0,1].plot(EC_inh_bin_edges, EC_inh_rate_filt_theta, linestyle='-', color='C1')
    axs4[0,1].plot(EC_inh_bin_edges, EC_inh_rate_filt_gamma, linestyle='-', color='C2')

    axs4[1,0].plot(DG_exc_bin_edges, DG_exc_rate, linestyle='-', color='b')
    axs4[1,0].plot(DG_exc_bin_edges, DG_exc_rate_filt_theta, linestyle='-', color='C1')
    axs4[1,0].plot(DG_exc_bin_edges, DG_exc_rate_filt_gamma, linestyle='-', color='C2')
    axs4[1,1].plot(DG_inh_bin_edges, DG_inh_rate, linestyle='-', color='r')
    axs4[1,1].plot(DG_inh_bin_edges, DG_inh_rate_filt_theta, linestyle='-', color='C1')
    axs4[1,1].plot(DG_inh_bin_edges, DG_inh_rate_filt_gamma, linestyle='-', color='C2')

    axs4[2,0].plot(CA3_exc_bin_edges, CA3_exc_rate, linestyle='-', color='b')
    axs4[2,0].plot(CA3_exc_bin_edges, CA3_exc_rate_filt_theta, linestyle='-', color='C1')
    axs4[2,0].plot(CA3_exc_bin_edges, CA3_exc_rate_filt_gamma, linestyle='-', color='C2')
    axs4[2,1].plot(CA3_inh_bin_edges, CA3_inh_rate, linestyle='-', color='r')
    axs4[2,1].plot(CA3_inh_bin_edges, CA3_inh_rate_filt_theta, linestyle='-', color='C1')
    axs4[2,1].plot(CA3_inh_bin_edges, CA3_inh_rate_filt_gamma, linestyle='-', color='C2')

    axs4[3,0].plot(CA1_exc_bin_edges, CA1_exc_rate, linestyle='-', color='b')
    axs4[3,0].plot(CA1_exc_bin_edges, CA1_exc_rate_filt_theta, linestyle='-', color='C1')
    axs4[3,0].plot(CA1_exc_bin_edges, CA1_exc_rate_filt_gamma, linestyle='-', color='C2')
    axs4[3,1].plot(CA1_inh_bin_edges, CA1_inh_rate, linestyle='-', color='r')
    axs4[3,1].plot(CA1_inh_bin_edges, CA1_inh_rate_filt_theta, linestyle='-', color='C1')
    axs4[3,1].plot(CA1_inh_bin_edges, CA1_inh_rate_filt_gamma, linestyle='-', color='C2')


    for ax_out in axs4:
        for ax_in in ax_out:
            ax_in.grid()

    # axs4[0,1].legend()
    plt.figlegend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    fig4.suptitle('Groups', fontsize=12)


    # Polar plots #1: theta vs gamma
    fig5, axs5 = plt.subplots(nrows=4, ncols=2, figsize=(8,12), subplot_kw={'projection': 'polar'})

    lags_exc = []
    PLV_exc = []
    lags_inh = []
    PLV_inh = []

    tlims = [1450, 1700]

    lags_tmp, PLV_tmp = calc_PLV(EC_exc_rate_filt_theta[tlims[0]:tlims[1]], EC_exc_rate_filt_gamma[tlims[0]:tlims[1]])
    lags_exc.append(lags_tmp)
    PLV_exc.append(PLV_tmp)
    lags_tmp, PLV_tmp = calc_PLV(DG_exc_rate_filt_theta[tlims[0]:tlims[1]], DG_exc_rate_filt_gamma[tlims[0]:tlims[1]])
    lags_exc.append(lags_tmp)
    PLV_exc.append(PLV_tmp)
    lags_tmp, PLV_tmp = calc_PLV(CA3_exc_rate_filt_theta[tlims[0]:tlims[1]], CA3_exc_rate_filt_gamma[tlims[0]:tlims[1]])
    lags_exc.append(lags_tmp)
    PLV_exc.append(PLV_tmp)
    lags_tmp, PLV_tmp = calc_PLV(CA1_exc_rate_filt_theta[tlims[0]:tlims[1]], CA1_exc_rate_filt_gamma[tlims[0]:tlims[1]])
    lags_exc.append(lags_tmp)
    PLV_exc.append(PLV_tmp)

    lags_tmp, PLV_tmp = calc_PLV(EC_inh_rate_filt_theta[tlims[0]:tlims[1]], EC_inh_rate_filt_gamma[tlims[0]:tlims[1]])
    lags_inh.append(lags_tmp)
    PLV_inh.append(PLV_tmp)
    lags_tmp, PLV_tmp = calc_PLV(DG_inh_rate_filt_theta[tlims[0]:tlims[1]], DG_inh_rate_filt_gamma[tlims[0]:tlims[1]])
    lags_inh.append(lags_tmp)
    PLV_inh.append(PLV_tmp)
    lags_tmp, PLV_tmp = calc_PLV(CA3_inh_rate_filt_theta[tlims[0]:tlims[1]], CA3_inh_rate_filt_gamma[tlims[0]:tlims[1]])
    lags_inh.append(lags_tmp)
    PLV_inh.append(PLV_tmp)
    lags_tmp, PLV_tmp = calc_PLV(CA1_inh_rate_filt_theta[tlims[0]:tlims[1]], CA1_inh_rate_filt_gamma[tlims[0]:tlims[1]])
    lags_inh.append(lags_tmp)
    PLV_inh.append(PLV_tmp)

    # E-groups
    for gidx in range(4):
        lags = lags_exc[gidx]
        PLV = PLV_exc[gidx]

        # Add black lines/arrows for a subset of phase lags
        for idx in np.arange(0, len(lags), 20):
            val = lags[idx]
            phi = np.angle(val)
            r = np.abs(val)
            # axs7.arrow(phi, 0., 0., r, )
            axs5[gidx,0].vlines(phi, 0., r, colors='k', alpha=0.2, zorder=3)

        # Red arrow for collective phase
        axs5[gidx,0].arrow(0., 0., PLV[1], PLV[0], alpha = 1., width = 0.015,
             edgecolor = 'red', facecolor = 'red', lw = 1, zorder = 4)

        axs5[gidx,0].set_ylim([0,1])
        axs5[gidx,0].set_title(r'PLV $\theta$ [{tl}-{th}]Hz - $\gamma$ [{gl}-{gh}]'.format(tl=fc_theta[0], th=fc_theta[1], gl=fc_gamma_low[0], gh=fc_gamma_low[1]))


    # I-groups
    for gidx in range(4):
        lags = lags_inh[gidx]
        PLV = PLV_inh[gidx]

        # Add black lines/arrows for a subset of phase lags
        for idx in np.arange(0, len(lags), 20):
            val = lags[idx]
            phi = np.angle(val)
            r = np.abs(val)
            # axs7.arrow(phi, 0., 0., r, )
            axs5[gidx,1].vlines(phi, 0., r, colors='k', alpha=0.2, zorder=3)

        # Red arrow for collective phase
        axs5[gidx,1].arrow(0., 0., PLV[1], PLV[0], alpha = 1., width = 0.015,
             edgecolor = 'red', facecolor = 'red', lw = 1, zorder = 4)

        axs5[gidx,1].set_ylim([0,1])
        axs5[gidx,1].set_title(r'PLV $\theta$ [{tl}-{th}]Hz - $\gamma$ [{gl}-{gh}]'.format(tl=fc_theta[0], th=fc_theta[1], gl=fc_gamma_low[0], gh=fc_gamma_low[1]))





    # Show the plots
    plt.tight_layout()
    plt.show()

    # Successful exit
    exit(0)
