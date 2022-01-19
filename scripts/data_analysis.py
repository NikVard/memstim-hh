#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import numpy as np
# import matplotlib.pyplot as plt

from brian2 import *
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy import signal


# Parameters
# ------------------------------------------------------------------------------
areas = ['EC', 'DG', 'CA3', 'CA1']
duration = 2*second
dt = 0.1*ms
fs = 1/dt
bin_size = 1*ms


# Directories
# ------------------------------------------------------------------------------
dirs = {}

# Results path
dirs['results'] = 'results/analysis/'
dirs['data'] = dirs['results'] + 'data/'

# Spikes
dirs['spikes'] = dirs['data'] + 'spikes/'


# Load the rasters
# ------------------------------------------------------------------------------
print(">> Loading rasters...")

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

EC_exc_N = 10000
EC_inh_N = 1000
DG_exc_N = 10000
DG_inh_N = 100
CA3_exc_N = 1000
CA3_inh_N = 100
CA1_exc_N = 10000
CA1_inh_N = 1000
N_tot = EC_exc_N + EC_inh_N + DG_exc_N + DG_inh_N + CA3_exc_N + CA3_inh_N + CA1_exc_N + CA1_inh_N


# Generate firing rates (LFP estimates)
# COBA example, brian2 docs
# https://brian2.readthedocs.io/en/stable/examples/frompapers.Stimberg_et_al_2018.example_1_COBA.html
# ------------------------------------------------------------------------------
print(">> Generating firing rates from rasters...")

EC_exc_spk_cnt, EC_exc_bin_edges = np.histogram(EC_exc_t, int(duration/bin_size))
EC_exc_rate = double(EC_exc_spk_cnt)/(EC_exc_N)/bin_size/Hz
EC_inh_spk_cnt, EC_inh_bin_edges = np.histogram(EC_inh_t, int(duration/bin_size))
EC_inh_rate = double(EC_inh_spk_cnt)/(EC_inh_N)/bin_size/Hz

DG_exc_spk_cnt, DG_exc_bin_edges = np.histogram(DG_exc_t, int(duration/bin_size))
DG_exc_rate = double(DG_exc_spk_cnt)/(DG_exc_N)/bin_size/Hz
DG_inh_spk_cnt, DG_inh_bin_edges = np.histogram(DG_inh_t, int(duration/bin_size))
DG_inh_rate = double(DG_inh_spk_cnt)/(DG_inh_N)/bin_size/Hz

CA3_exc_spk_cnt, CA3_exc_bin_edges = np.histogram(CA3_exc_t, int(duration/bin_size))
CA3_exc_rate = double(CA3_exc_spk_cnt)/(CA3_exc_N)/bin_size/Hz
CA3_inh_spk_cnt, CA3_inh_bin_edges = np.histogram(CA3_inh_t, int(duration/bin_size))
CA3_inh_rate = double(CA3_inh_spk_cnt)/(CA3_inh_N)/bin_size/Hz

CA1_exc_spk_cnt, CA1_exc_bin_edges = np.histogram(CA1_exc_t, int(duration/bin_size))
CA1_exc_rate = double(CA1_exc_spk_cnt)/(CA1_exc_N)/bin_size/Hz
CA1_inh_spk_cnt, CA1_inh_bin_edges = np.histogram(CA1_inh_t, int(duration/bin_size))
CA1_inh_rate = double(CA1_inh_spk_cnt)/(CA1_inh_N)/bin_size/Hz


# Detrend (detrend) the signals
# ------------------------------------------------------------------------------
print(">> Detrending signals...")
EC_exc_rate_dtr = detrend(EC_exc_rate)
EC_inh_rate_dtr = detrend(EC_inh_rate)
DG_exc_rate_dtr = detrend(DG_exc_rate)
DG_inh_rate_dtr = detrend(DG_inh_rate)
CA3_exc_rate_dtr = detrend(CA3_exc_rate)
CA3_inh_rate_dtr = detrend(CA3_inh_rate)
CA1_exc_rate_dtr = detrend(CA1_exc_rate)
CA1_inh_rate_dtr = detrend(CA1_inh_rate)


# Plot the firing rates
# ------------------------------------------------------------------------------
print(">> Plotting firing rates...")
fig1, axs1 = plt.subplots(4, sharex=True, sharey=True, figsize=(16,8))
axs1[0].plot(EC_exc_bin_edges[:-1], EC_exc_rate, linestyle='-', color='b', label= 'exc')
axs1[0].plot(EC_inh_bin_edges[:-1], EC_inh_rate, linestyle='--', color='r', label='inh')
axs1[1].plot(DG_exc_bin_edges[:-1], DG_exc_rate, linestyle='-', color='b', label='exc')
axs1[1].plot(DG_inh_bin_edges[:-1], DG_inh_rate, linestyle='--', color='r', label='inh')
axs1[2].plot(CA3_exc_bin_edges[:-1], CA3_exc_rate, linestyle='-', color='b', label='exc')
axs1[2].plot(CA3_inh_bin_edges[:-1], CA3_inh_rate, linestyle='--', color='r', label='inh')
axs1[3].plot(CA1_exc_bin_edges[:-1], CA1_exc_rate, linestyle='-', color='b', label='exc')
axs1[3].plot(CA1_inh_bin_edges[:-1], CA1_inh_rate, linestyle='--', color='r', label='inh')

for (ax, lbl) in zip(axs1, areas):
    ax.set_title(lbl)
    ax.set_ylabel('Rate [Hz]', rotation=0, fontsize=12, labelpad=30)
    ax.grid(linestyle='-', color=[0.5, 0.5, 0.5], alpha=0.33)
    ax.set_ylim([0, 1000])

axs1[3].set_xlabel('Time [ms]')
axs1[3].legend()

# plt.show()


# Analysis #1: Frequency (PSD)
# ------------------------------------------------------------------------------
from scipy.fft import rfft, rfftfreq

print("\n* Analysis #1")
print(">> Calculating PSDs...")

win = 'boxcar' # periodogram default
# win = 'hann'
N = len(EC_exc_rate)
NFFT = int(pow(2, ceil(log(2*N)/log(2))))
dt2 = bin_size
fs2 = 1/dt2

# Compute the power spectral density
# TODO: Switch to rfft/rfftfreq and compute PSD as abs(X)^2
# f, EC_exc_Pxx = signal.periodogram(EC_exc_rate, fs2, window=win)
# EC_inh_Pxx = signal.periodogram(EC_inh_rate, fs2, window=win)[1]
# DG_exc_Pxx = signal.periodogram(DG_exc_rate, fs2, window=win)[1]
# DG_inh_Pxx = signal.periodogram(DG_inh_rate, fs2, window=win)[1]
# CA3_exc_Pxx = signal.periodogram(CA3_exc_rate, fs2, window=win)[1]
# CA3_inh_Pxx = signal.periodogram(CA3_inh_rate, fs2, window=win)[1]
# CA1_exc_Pxx = signal.periodogram(CA1_exc_rate, fs2, window=win)[1]
# CA1_inh_Pxx = signal.periodogram(CA1_inh_rate, fs2, window=win)[1]

f = rfftfreq(NFFT, dt2)
EC_exc_Pxx = np.abs(rfft(EC_exc_rate_dtr, n=NFFT)/NFFT)**2
EC_inh_Pxx = np.abs(rfft(EC_inh_rate_dtr, n=NFFT)/NFFT)**2
DG_exc_Pxx = np.abs(rfft(DG_exc_rate_dtr, n=NFFT)/NFFT)**2
DG_inh_Pxx = np.abs(rfft(DG_inh_rate_dtr, n=NFFT)/NFFT)**2
CA3_exc_Pxx = np.abs(rfft(CA3_exc_rate_dtr, n=NFFT)/NFFT)**2
CA3_inh_Pxx = np.abs(rfft(CA3_inh_rate_dtr, n=NFFT)/NFFT)**2
CA1_exc_Pxx = np.abs(rfft(CA1_exc_rate_dtr, n=NFFT)/NFFT)**2
CA1_inh_Pxx = np.abs(rfft(CA1_inh_rate_dtr, n=NFFT)/NFFT)**2


# Plot the PSDs
# ------------------------------------------------------------------------------
fig2, axs2 = plt.subplots(4, sharex=True, sharey=True, figsize=(16,8))
axs2[0].plot(f, EC_exc_Pxx, linestyle='-', color='b', label='exc')
axs2[0].plot(f, EC_inh_Pxx, linestyle='--', color='r', label='inh')
axs2[1].plot(f, DG_exc_Pxx, linestyle='-', color='b', label='exc')
axs2[1].plot(f, DG_inh_Pxx, linestyle='--', color='r', label='inh')
axs2[2].plot(f, CA3_exc_Pxx, linestyle='-', color='b', label='exc')
axs2[2].plot(f, CA3_inh_Pxx, linestyle='--', color='r', label='inh')
axs2[3].plot(f, CA1_exc_Pxx, linestyle='-', color='b', label='exc')
axs2[3].plot(f, CA1_inh_Pxx, linestyle='--', color='r', label='inh')

# axs2[0].plot(f, np.log10(EC_exc_Pxx), linestyle='-', color='b', label='exc')
# axs2[0].plot(f, np.log10(EC_inh_Pxx), linestyle='--', color='r', label='inh')
# axs2[1].plot(f, np.log10(DG_exc_Pxx), linestyle='-', color='b', label='exc')
# axs2[1].plot(f, np.log10(DG_inh_Pxx), linestyle='--', color='r', label='inh')
# axs2[2].plot(f, np.log10(CA3_exc_Pxx), linestyle='-', color='b', label='exc')
# axs2[2].plot(f, np.log10(CA3_inh_Pxx), linestyle='--', color='r', label='inh')
# axs2[3].plot(f, np.log10(CA1_exc_Pxx), linestyle='-', color='b', label='exc')
# axs2[3].plot(f, np.log10(CA1_inh_Pxx), linestyle='--', color='r', label='inh')



# axs2[0].semilogx(f, EC_exc_Pxx, linestyle='-', color='b', label='exc')
# axs2[0].semilogx(f, EC_inh_Pxx, linestyle='--', color='r', label='inh')
# axs2[1].semilogx(f, DG_exc_Pxx, linestyle='-', color='b', label='exc')
# axs2[1].semilogx(f, DG_inh_Pxx, linestyle='--', color='r', label='inh')
# axs2[2].semilogx(f, CA3_exc_Pxx, linestyle='-', color='b', label='exc')
# axs2[2].semilogx(f, CA3_inh_Pxx, linestyle='--', color='r', label='inh')
# axs2[3].semilogx(f, CA1_exc_Pxx, linestyle='-', color='b', label='exc')
# axs2[3].semilogx(f, CA1_inh_Pxx, linestyle='--', color='r', label='inh')

# axs2[0].semilogy(f, EC_exc_Pxx, linestyle='-', color='b', label='exc')
# axs2[0].semilogy(f, EC_inh_Pxx, linestyle='--', color='r', label='inh')
# axs2[1].semilogy(f, DG_exc_Pxx, linestyle='-', color='b', label='exc')
# axs2[1].semilogy(f, DG_inh_Pxx, linestyle='--', color='r', label='inh')
# axs2[2].semilogy(f, CA3_exc_Pxx, linestyle='-', color='b', label='exc')
# axs2[2].semilogy(f, CA3_inh_Pxx, linestyle='--', color='r', label='inh')
# axs2[3].semilogy(f, CA1_exc_Pxx, linestyle='-', color='b', label='exc')
# axs2[3].semilogy(f, CA1_inh_Pxx, linestyle='--', color='r', label='inh')

# axs2[0].set_xscale('log')
# axs2[1].set_xscale('log')
# axs2[2].set_xscale('log')
# axs2[3].set_xscale('log')

# axs2[0].set_yscale('log')
# axs2[1].set_yscale('log')
# axs2[2].set_yscale('log')
# axs2[3].set_yscale('log')

for (ax, lbl) in zip(axs2, areas):
    ax.set_title(lbl)
    ax.set_ylabel(r'$\frac{V^2}{Hz}$', rotation=0, fontsize=16, labelpad=20)
    ax.grid(linestyle='-', color=[0.5, 0.5, 0.5], alpha=0.33)
    # ax.set_xlim([10e-1,100])
    # ax.set_ylim([0, 100])

axs2[3].set_xlabel('Frequency [Hz]')
axs2[3].legend()

# plt.show()
# exit()


# Analysis #2a: Time-Frequency % TODO: Fix Z-limits and add colorbar
# ------------------------------------------------------------------------------
print("\n* Analysis #2a")
print(">> Time-frequency analysis (scipy spectrogram)...")

win_size_t = 0.25
win_size_s = int(win_size_t * fs2)

f, t, EC_exc_Sxx = signal.spectrogram(EC_exc_rate_dtr, fs=int(fs2), nfft=NFFT, window=signal.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))
EC_inh_Sxx = signal.spectrogram(EC_inh_rate_dtr, fs=int(fs2), nfft=NFFT, window=signal.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
DG_exc_Sxx = signal.spectrogram(DG_exc_rate_dtr, fs=int(fs2), nfft=NFFT, window=signal.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
DG_inh_Sxx = signal.spectrogram(DG_inh_rate_dtr, fs=int(fs2), nfft=NFFT, window=signal.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
CA3_exc_Sxx = signal.spectrogram(CA3_exc_rate_dtr, fs=int(fs2), nfft=NFFT, window=signal.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
CA3_inh_Sxx = signal.spectrogram(CA3_inh_rate_dtr, fs=int(fs2), nfft=NFFT, window=signal.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
CA1_exc_Sxx = signal.spectrogram(CA1_exc_rate_dtr, fs=int(fs2), nfft=NFFT, window=signal.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]
CA1_inh_Sxx = signal.spectrogram(CA1_inh_rate_dtr, fs=int(fs2), nfft=NFFT, window=signal.hann(M=win_size_s,sym=False), nperseg=win_size_s, noverlap=int(2*win_size_s/4))[2]


fig10, axs10 = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(16,8))

axs10[0,0].pcolormesh(t, f, 1./win_size_s*EC_exc_Sxx**2, shading='gouraud')
axs10[1,0].pcolormesh(t, f, 1./win_size_s*EC_inh_Sxx**2, shading='gouraud')
axs10[0,1].pcolormesh(t, f, 1./win_size_s*DG_exc_Sxx**2, shading='gouraud')
axs10[1,1].pcolormesh(t, f, 1./win_size_s*DG_inh_Sxx**2, shading='gouraud')
axs10[0,2].pcolormesh(t, f, 1./win_size_s*CA3_exc_Sxx**2, shading='gouraud')
axs10[1,2].pcolormesh(t, f, 1./win_size_s*CA3_inh_Sxx**2, shading='gouraud')
axs10[0,3].pcolormesh(t, f, 1./win_size_s*CA1_exc_Sxx**2, shading='gouraud')
axs10[1,3].pcolormesh(t, f, 1./win_size_s*CA1_inh_Sxx**2, shading='gouraud')


for ax_out in axs10:
    for ax_in in ax_out:
        ax_in.set_ylabel('Frequency [Hz]')
        ax_in.set_xlabel('Time [sec]')
        ax_in.set_ylim(0,100)

axs10[0,0].set_title('EC')
axs10[0,1].set_title('DG')
axs10[0,2].set_title('CA3')
axs10[0,3].set_title('CA1')

fig10.suptitle('Spectrograms', fontsize=16)

# plt.show()
# exit()


# Analysis #2b: Time-Frequency (wavelets?)
# ------------------------------------------------------------------------------
print("\n* Analysis #2b")
print(">> Time-frequency analysis (spectrogram) using wavelets...")
print(":: PASS ::")


# Analysis #3: PAC of theta-gamma rhythms
# https://www.frontiersin.org/articles/10.3389/fnins.2019.00573/full#:~:text=To%20calculate%20phase%2Damplitude%20coupling%2C%20first%2C%20raw%20data%20is,the%20complex%2Dvalued%20analytic%20signal.
# ------------------------------------------------------------------------------
print("\n* Analysis #3")
print(">> Phase-amplitude coupling in theta-gamma using filter-Hilbert method...")

def butter_lowpass(lowcut, fs, order=8, sos=False):
    ''' Create a lowpass butterworth filter '''
    nyq = 0.5 * fs
    low = lowcut / nyq

    if sos:
        sos_out = signal.butter(order, low, analog=False, btype='low', output='sos')
        return sos_out

    b, a = signal.butter(order, low, analog=False, btype='low', output='ba')
    return b, a

def butter_highpass(highcut, fs, order=8, sos=False):
    ''' Create a highpass butterworth filter '''
    nyq = 0.5 * fs
    high = highcut / nyq

    if sos:
        sos_out = signal.butter(order, high, analog=False, btype='high', output='sos')
        return sos_out

    b, a = signal.butter(order, high, analog=False, btype='high', output='ba')
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=8, sos=False):
    ''' Create a bandpass butterworth filter '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if sos:
        sos_out = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos_out

    b, a = signal.butter(order, [low, high], analog=False, btype='band', output='ba')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order=5, sos=False):
    ''' Lowpass filter the data '''
    if sos:
        sos_out = butter_lowpass(lowcut, fs, order=order, sos=sos)
        y = signal.sosfiltfilt(sos_out, data)
    else:
        b, a = butter_lowpass(lowcut, fs, order=order, sos=sos)
        y = signal.filtfilt(b, a, data)

    return y

def butter_highpass_filter(data, highcut, fs, order=5, sos=False):
    ''' Highpass filter the data '''
    if sos:
        sos_out = butter_highpass(highcut, fs, order=order, sos=sos)
        y = signal.sosfiltfilt(sos_out, data)
    else:
        b, a = butter_highpass(highcut, fs, order=order, sos=sos)
        y = signal.filtfilt(b, a, data)

    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, sos=False):
    ''' Bandpass filter the data '''
    if sos:
        sos_out = butter_bandpass(lowcut, highcut, fs, order=order, sos=sos)
        y = signal.sosfiltfilt(sos_out, data)
    else:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order, sos=sos)
        y = signal.filtfilt(b, a, data)

    return y

# Make butterworth n-th order bandpass filters for theta/gamma bands
# Theta band filter
N = 12 # filter order
fc_theta = [4, 10] # theta frequencies [Hz]
fc_gamma_low = [30, 80] # low-gamma frequencies [Hz]
fc_gamma_high = [80, 120] # high-gamma frequencies [Hz]

# Theta band filter
b, a = butter_bandpass(fc_theta[0], fc_theta[1], fs=int(fs2), order=N, sos=False)
sos = butter_bandpass(fc_theta[0], fc_theta[1], fs=int(fs2), order=N, sos=True)
filt_theta = {'a':a, 'b':b, 'sos':sos}

# Low-gamma filter
b, a = butter_bandpass(fc_gamma_low[0], fc_gamma_low[1], fs=int(fs2), order=N, sos=False)
sos = butter_bandpass(fc_gamma_low[0], fc_gamma_low[1], fs=int(fs2), order=N, sos=True)
filt_gamma_low = {'a':a,'b':b, 'sos':sos}

# High-gamma filter
b, a = butter_bandpass(fc_gamma_high[0], fc_gamma_high[1], fs=int(fs2), order=N, sos=False)
sos = butter_bandpass(fc_gamma_high[0], fc_gamma_high[1], fs=int(fs2), order=N, sos=True)
filt_gamma_high = {'a':a,'b':b, 'sos':sos}


# Test the filters
# EC
data = EC_exc_rate_dtr
EC_exc_rate_filt_low = butter_lowpass_filter(data, lowcut=4, fs=int(fs2), order=N, sos=True)
EC_exc_rate_filt_theta = butter_bandpass_filter(data, lowcut=fc_theta[0], highcut=fc_theta[1], fs=int(fs2), order=N, sos=True)
EC_exc_rate_filt_gamma_low = butter_bandpass_filter(data, lowcut=fc_gamma_low[0], highcut=fc_gamma_low[1], fs=int(fs2), order=N, sos=True)
EC_exc_rate_filt_gamma_high = butter_bandpass_filter(data, lowcut=fc_gamma_high[0], highcut=fc_gamma_high[1], fs=int(fs2), order=N, sos=True)
EC_exc_rate_filt_high = butter_highpass_filter(data, highcut=120, fs=int(fs2), order=N, sos=True)

data = EC_inh_rate_dtr
EC_inh_rate_filt_low = butter_lowpass_filter(data, lowcut=4, fs=int(fs2), order=N, sos=True)
EC_inh_rate_filt_theta = butter_bandpass_filter(data, lowcut=fc_theta[0], highcut=fc_theta[1], fs=int(fs2), order=N, sos=True)
EC_inh_rate_filt_gamma_low = butter_bandpass_filter(data, lowcut=fc_gamma_low[0], highcut=fc_gamma_low[1], fs=int(fs2), order=N, sos=True)
EC_inh_rate_filt_gamma_high = butter_bandpass_filter(data, lowcut=fc_gamma_high[0], highcut=fc_gamma_high[1], fs=int(fs2), order=N, sos=True)
EC_inh_rate_filt_high = butter_highpass_filter(data, highcut=120, fs=int(fs2), order=N, sos=True)

# DG
data = DG_exc_rate_dtr
DG_exc_rate_filt_low = butter_lowpass_filter(data, lowcut=4, fs=int(fs2), order=N, sos=True)
DG_exc_rate_filt_theta = butter_bandpass_filter(data, lowcut=fc_theta[0], highcut=fc_theta[1], fs=int(fs2), order=N, sos=True)
DG_exc_rate_filt_gamma_low = butter_bandpass_filter(data, lowcut=fc_gamma_low[0], highcut=fc_gamma_low[1], fs=int(fs2), order=N, sos=True)
DG_exc_rate_filt_gamma_high = butter_bandpass_filter(data, lowcut=fc_gamma_high[0], highcut=fc_gamma_high[1], fs=int(fs2), order=N, sos=True)
DG_exc_rate_filt_high = butter_highpass_filter(data, highcut=120, fs=int(fs2), order=N, sos=True)

data = DG_inh_rate_dtr
DG_inh_rate_filt_low = butter_lowpass_filter(data, lowcut=4, fs=int(fs2), order=N, sos=True)
DG_inh_rate_filt_theta = butter_bandpass_filter(data, lowcut=fc_theta[0], highcut=fc_theta[1], fs=int(fs2), order=N, sos=True)
DG_inh_rate_filt_gamma_low = butter_bandpass_filter(data, lowcut=fc_gamma_low[0], highcut=fc_gamma_low[1], fs=int(fs2), order=N, sos=True)
DG_inh_rate_filt_gamma_high = butter_bandpass_filter(data, lowcut=fc_gamma_high[0], highcut=fc_gamma_high[1], fs=int(fs2), order=N, sos=True)
DG_inh_rate_filt_high = butter_highpass_filter(data, highcut=120, fs=int(fs2), order=N, sos=True)

# CA3
data = CA3_exc_rate_dtr
CA3_exc_rate_filt_low = butter_lowpass_filter(data, lowcut=4, fs=int(fs2), order=N, sos=True)
CA3_exc_rate_filt_theta = butter_bandpass_filter(data, lowcut=fc_theta[0], highcut=fc_theta[1], fs=int(fs2), order=N, sos=True)
CA3_exc_rate_filt_gamma_low = butter_bandpass_filter(data, lowcut=fc_gamma_low[0], highcut=fc_gamma_low[1], fs=int(fs2), order=N, sos=True)
CA3_exc_rate_filt_gamma_high = butter_bandpass_filter(data, lowcut=fc_gamma_high[0], highcut=fc_gamma_high[1], fs=int(fs2), order=N, sos=True)
CA3_exc_rate_filt_high = butter_highpass_filter(data, highcut=120, fs=int(fs2), order=N, sos=True)

data = CA3_inh_rate_dtr
CA3_inh_rate_filt_low = butter_lowpass_filter(data, lowcut=4, fs=int(fs2), order=N, sos=True)
CA3_inh_rate_filt_theta = butter_bandpass_filter(data, lowcut=fc_theta[0], highcut=fc_theta[1], fs=int(fs2), order=N, sos=True)
CA3_inh_rate_filt_gamma_low = butter_bandpass_filter(data, lowcut=fc_gamma_low[0], highcut=fc_gamma_low[1], fs=int(fs2), order=N, sos=True)
CA3_inh_rate_filt_gamma_high = butter_bandpass_filter(data, lowcut=fc_gamma_high[0], highcut=fc_gamma_high[1], fs=int(fs2), order=N, sos=True)
CA3_inh_rate_filt_high = butter_highpass_filter(data, highcut=120, fs=int(fs2), order=N, sos=True)

# CA1
data = CA1_exc_rate_dtr
CA1_exc_rate_filt_low = butter_lowpass_filter(data, lowcut=4, fs=int(fs2), order=N, sos=True)
CA1_exc_rate_filt_theta = butter_bandpass_filter(data, lowcut=fc_theta[0], highcut=fc_theta[1], fs=int(fs2), order=N, sos=True)
CA1_exc_rate_filt_gamma_low = butter_bandpass_filter(data, lowcut=fc_gamma_low[0], highcut=fc_gamma_low[1], fs=int(fs2), order=N, sos=True)
CA1_exc_rate_filt_gamma_high = butter_bandpass_filter(data, lowcut=fc_gamma_high[0], highcut=fc_gamma_high[1], fs=int(fs2), order=N, sos=True)
CA1_exc_rate_filt_high = butter_highpass_filter(data, highcut=120, fs=int(fs2), order=N, sos=True)

data = CA1_inh_rate_dtr
CA1_inh_rate_filt_low = butter_lowpass_filter(data, lowcut=4, fs=int(fs2), order=N, sos=True)
CA1_inh_rate_filt_theta = butter_bandpass_filter(data, lowcut=fc_theta[0], highcut=fc_theta[1], fs=int(fs2), order=N, sos=True)
CA1_inh_rate_filt_gamma_low = butter_bandpass_filter(data, lowcut=fc_gamma_low[0], highcut=fc_gamma_low[1], fs=int(fs2), order=N, sos=True)
CA1_inh_rate_filt_gamma_high = butter_bandpass_filter(data, lowcut=fc_gamma_high[0], highcut=fc_gamma_high[1], fs=int(fs2), order=N, sos=True)
CA1_inh_rate_filt_high = butter_highpass_filter(data, highcut=120, fs=int(fs2), order=N, sos=True)


# Plot excitatory groups
fig4, axs4 = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(16,8))
axs4[0,0].plot(EC_exc_bin_edges[:-1], EC_exc_rate_dtr, label='raw')
axs4[1,0].plot(EC_exc_bin_edges[:-1], EC_exc_rate_filt_low, color='C0', ls='--', label=r'<4Hz', zorder=1)
axs4[1,0].plot(EC_exc_bin_edges[:-1], EC_exc_rate_filt_theta, color='C1', ls='--', label=r'$\theta$', zorder=4)
axs4[1,0].plot(EC_exc_bin_edges[:-1], EC_exc_rate_filt_gamma_low, color='C2', ls='--', label=r'low-$\gamma$', zorder=3)
axs4[1,0].plot(EC_exc_bin_edges[:-1], EC_exc_rate_filt_gamma_high, color='C3', ls='--', label=r'high-$\gamma$', zorder=2)
axs4[1,0].plot(EC_exc_bin_edges[:-1], EC_exc_rate_filt_gamma_high, color='C4', ls='--', label=r'low-$\gamma$', zorder=0)

axs4[0,1].plot(DG_exc_bin_edges[:-1], DG_exc_rate_dtr)
axs4[1,1].plot(DG_exc_bin_edges[:-1], DG_exc_rate_filt_low, color='C0', ls='--', zorder=1)
axs4[1,1].plot(DG_exc_bin_edges[:-1], DG_exc_rate_filt_theta, color='C1', ls='--', zorder=4)
axs4[1,1].plot(DG_exc_bin_edges[:-1], DG_exc_rate_filt_gamma_low, color='C2', ls='--', zorder=3)
axs4[1,1].plot(DG_exc_bin_edges[:-1], DG_exc_rate_filt_gamma_high, color='C3', ls='--', zorder=2)
axs4[1,1].plot(DG_exc_bin_edges[:-1], DG_exc_rate_filt_high, color='C4', ls='--', zorder=0)

axs4[0,2].plot(CA3_exc_bin_edges[:-1], CA3_exc_rate_dtr)
axs4[1,2].plot(CA3_exc_bin_edges[:-1], CA3_exc_rate_filt_low, color='C0', ls='--', zorder=1)
axs4[1,2].plot(CA3_exc_bin_edges[:-1], CA3_exc_rate_filt_theta, color='C1', ls='--', zorder=4)
axs4[1,2].plot(CA3_exc_bin_edges[:-1], CA3_exc_rate_filt_gamma_low, color='C2', ls='--', zorder=3)
axs4[1,2].plot(CA3_exc_bin_edges[:-1], CA3_exc_rate_filt_gamma_high, color='C3', ls='--', zorder=2)
axs4[1,2].plot(CA3_exc_bin_edges[:-1], CA3_exc_rate_filt_high, color='C4', ls='--', zorder=0)

axs4[0,3].plot(CA1_exc_bin_edges[:-1], CA1_exc_rate_dtr)
axs4[1,3].plot(CA1_exc_bin_edges[:-1], CA1_exc_rate_filt_low, color='C0', ls='--', zorder=1)
axs4[1,3].plot(CA1_exc_bin_edges[:-1], CA1_exc_rate_filt_theta, color='C1', ls='--', zorder=4)
axs4[1,3].plot(CA1_exc_bin_edges[:-1], CA1_exc_rate_filt_gamma_low, color='C2', ls='--', zorder=3)
axs4[1,3].plot(CA1_exc_bin_edges[:-1], CA1_exc_rate_filt_gamma_high, color='C3', ls='--', zorder=2)
axs4[1,3].plot(CA1_exc_bin_edges[:-1], CA1_exc_rate_filt_high, color='C4', ls='--', zorder=0)

axs4[0,0].set_title('EC')
axs4[0,1].set_title('DG')
axs4[0,2].set_title('CA3')
axs4[0,3].set_title('CA1')

axs4[1,0].set_xlabel('Time [ms]')
axs4[1,1].set_xlabel('Time [ms]')
axs4[1,2].set_xlabel('Time [ms]')
axs4[1,3].set_xlabel('Time [ms]')

# axs4[0,0].set_ylim(bottom=0.)

for ax_out in axs4:
    for ax_in in ax_out:
        ax_in.grid()

# axs4[0,1].legend()
plt.figlegend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig4.suptitle('Excitatory Groups', fontsize=16)

# Plot inhibitory groups
fig5, axs5 = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(16,8))
axs5[0,0].plot(EC_inh_bin_edges[:-1], EC_inh_rate_dtr, label='raw')
axs5[1,0].plot(EC_inh_bin_edges[:-1], EC_inh_rate_filt_low, color='C0', ls='--', label=r'$f_c < 4$ Hz', zorder=1)
axs5[1,0].plot(EC_inh_bin_edges[:-1], EC_inh_rate_filt_theta, color='C1', ls='--', label=r'$\theta$', zorder=4)
axs5[1,0].plot(EC_inh_bin_edges[:-1], EC_inh_rate_filt_gamma_low, color='C2', ls='--', label=r'low-$\gamma$', zorder=3)
axs5[1,0].plot(EC_inh_bin_edges[:-1], EC_inh_rate_filt_gamma_high, color='C3', ls='--', label=r'high-$\gamma$', zorder=2)
axs5[1,0].plot(EC_inh_bin_edges[:-1], EC_inh_rate_filt_high, color='C4', ls='--', label=r'$f_c > 120$ Hz', zorder=0)

axs5[0,1].plot(DG_inh_bin_edges[:-1], DG_inh_rate_dtr)
axs5[1,1].plot(DG_inh_bin_edges[:-1], DG_inh_rate_filt_low, color='C0', ls='--', zorder=1)
axs5[1,1].plot(DG_inh_bin_edges[:-1], DG_inh_rate_filt_theta, color='C1', ls='--', zorder=4)
axs5[1,1].plot(DG_inh_bin_edges[:-1], DG_inh_rate_filt_gamma_low, color='C2', ls='--', zorder=3)
axs5[1,1].plot(DG_inh_bin_edges[:-1], DG_inh_rate_filt_gamma_high, color='C3', ls='--', zorder=2)
axs5[1,1].plot(DG_inh_bin_edges[:-1], DG_inh_rate_filt_high, color='C4', ls='--', zorder=0)

axs5[0,2].plot(CA3_inh_bin_edges[:-1], CA3_inh_rate_dtr)
axs5[1,2].plot(CA3_inh_bin_edges[:-1], CA3_inh_rate_filt_low, color='C0', ls='--', zorder=1)
axs5[1,2].plot(CA3_inh_bin_edges[:-1], CA3_inh_rate_filt_theta, color='C1', ls='--', zorder=4)
axs5[1,2].plot(CA3_inh_bin_edges[:-1], CA3_inh_rate_filt_gamma_low, color='C2', ls='--', zorder=3)
axs5[1,2].plot(CA3_inh_bin_edges[:-1], CA3_inh_rate_filt_gamma_high, color='C3', ls='--', zorder=2)
axs5[1,2].plot(CA3_inh_bin_edges[:-1], CA3_inh_rate_filt_high, color='C4', ls='--', zorder=0)

axs5[0,3].plot(CA1_inh_bin_edges[:-1], CA1_inh_rate_dtr)
axs5[1,3].plot(CA1_inh_bin_edges[:-1], CA1_inh_rate_filt_low, color='C0', ls='--', zorder=1)
axs5[1,3].plot(CA1_inh_bin_edges[:-1], CA1_inh_rate_filt_theta, color='C1', ls='--', zorder=4)
axs5[1,3].plot(CA1_inh_bin_edges[:-1], CA1_inh_rate_filt_gamma_low, color='C2', ls='--', zorder=3)
axs5[1,3].plot(CA1_inh_bin_edges[:-1], CA1_inh_rate_filt_gamma_high, color='C3', ls='--', zorder=2)
axs5[1,3].plot(CA1_inh_bin_edges[:-1], CA1_inh_rate_filt_high, color='C4', ls='--', zorder=0)

axs5[0,0].set_title('EC')
axs5[0,1].set_title('DG')
axs5[0,2].set_title('CA3')
axs5[0,3].set_title('CA1')

axs5[1,0].set_xlabel('Time [ms]')
axs5[1,1].set_xlabel('Time [ms]')
axs5[1,2].set_xlabel('Time [ms]')
axs5[1,3].set_xlabel('Time [ms]')

# axs4[0,0].set_ylim(bottom=0.)

for ax_out in axs5:
    for ax_in in ax_out:
        ax_in.grid()

# axs4[0,1].legend()
plt.figlegend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig5.suptitle('Inhibitory Groups', fontsize=16)


# plt.show()

# Analytic signals from Hilbert transform
# Theta
EC_exc_theta_a = signal.hilbert(EC_exc_rate_filt_theta)
EC_inh_theta_a = signal.hilbert(EC_inh_rate_filt_theta)
DG_exc_theta_a = signal.hilbert(DG_exc_rate_filt_theta)
DG_inh_theta_a = signal.hilbert(DG_inh_rate_filt_theta)
CA3_exc_theta_a = signal.hilbert(CA3_exc_rate_filt_theta)
CA3_inh_theta_a = signal.hilbert(CA3_inh_rate_filt_theta)
CA1_exc_theta_a = signal.hilbert(CA1_exc_rate_filt_theta)
CA1_inh_theta_a = signal.hilbert(CA1_inh_rate_filt_theta)

# Gamma (low)
EC_exc_gamma_low_a = signal.hilbert(EC_exc_rate_filt_gamma_low)
EC_inh_gamma_low_a = signal.hilbert(EC_inh_rate_filt_gamma_low)
DG_exc_gamma_low_a = signal.hilbert(DG_exc_rate_filt_gamma_low)
DG_inh_gamma_low_a = signal.hilbert(DG_inh_rate_filt_gamma_low)
CA3_exc_gamma_low_a = signal.hilbert(CA3_exc_rate_filt_gamma_low)
CA3_inh_gamma_low_a = signal.hilbert(CA3_inh_rate_filt_gamma_low)
CA1_exc_gamma_low_a = signal.hilbert(CA1_exc_rate_filt_gamma_low)
CA1_inh_gamma_low_a = signal.hilbert(CA1_inh_rate_filt_gamma_low)

# Gamma (high)
EC_exc_gamma_high_a = signal.hilbert(EC_exc_rate_filt_gamma_high)
EC_inh_gamma_high_a = signal.hilbert(EC_inh_rate_filt_gamma_high)
DG_exc_gamma_high_a = signal.hilbert(DG_exc_rate_filt_gamma_high)
DG_inh_gamma_high_a = signal.hilbert(DG_inh_rate_filt_gamma_high)
CA3_exc_gamma_high_a = signal.hilbert(CA3_exc_rate_filt_gamma_high)
CA3_inh_gamma_high_a = signal.hilbert(CA3_inh_rate_filt_gamma_high)
CA1_exc_gamma_high_a = signal.hilbert(CA1_exc_rate_filt_gamma_high)
CA1_inh_gamma_high_a = signal.hilbert(CA1_inh_rate_filt_gamma_high)


fig6, axs6 = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(16,8))
axs6[0,0].plot(EC_exc_bin_edges[:-1], EC_exc_rate_filt_theta, label=r'filtered $\theta$')
axs6[0,0].plot(EC_exc_bin_edges[:-1], np.abs(EC_exc_theta_a), label=r'envelope')
axs6[1,0].plot(EC_exc_bin_edges[:-1], np.angle(EC_exc_theta_a), label=r'phase')

axs6[0,1].plot(EC_exc_bin_edges[:-1], EC_exc_rate_filt_gamma_low, label=r'filtered $\gamma_{0}$')
axs6[0,1].plot(EC_exc_bin_edges[:-1], np.abs(EC_exc_gamma_low_a), label=r'envelope')
axs6[1,1].plot(EC_exc_bin_edges[:-1], np.angle(EC_exc_gamma_low_a), label=r'phase')

axs6[1,0].set_ylabel('Phase')
axs6[0,0].set_ylabel('Amplitude')
axs6[1,0].set_xlabel('Time [ms]')
axs6[1,1].set_xlabel('Time [ms]')

# plt.figlegend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig6.suptitle('EC Excitatory Hilbert Transform Test', fontsize=16)

# plt.show()

# PAC calculation: use either the Phase-Locking-Value (PLV - Mormann 2005) or the Mean Vector Length (MVL - Canolty 2006) (most specific); by the MI as last resort

# PL: Phase extracted from low-frequency (Theta); amplitude extracted from high-frequency (Gamma)
# The amplitude time series is then again Hilbert transformed and phase is extracted from the “second” analytic signal
# By these steps, one obtains phase angles for both time series for each data point.


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
    sigl_a = signal.hilbert(sigl)

    # 2. Hilbert-transform higher-frequency signal
    sigh_a = signal.hilbert(sigh)

    # 3. Hilbert-transform the amplitude of the HF-analytic signal
    sigh_aa = signal.hilbert(np.real(sigh_a))

    # 4. Calculate the phase differences
    phi_lt = np.angle(sigl_a)
    phi_ht = np.angle(sigh_aa)
    phi_d = phi_lt - phi_ht
    lags = np.exp(1j*phi_d)

    # 5. Calculate the PLV (mean of lags)
    mu_phi = np.mean(lags)
    PLV = (np.abs(mu_phi), np.angle(mu_phi))

    return lags, PLV


# Polar plot #1: theta vs low-gamma
# Test the PLV
lags, PLV = calc_PLV(EC_exc_rate_filt_theta, EC_exc_rate_filt_gamma_low)
fig7, axs7 = plt.subplots(1, figsize=(12,12), subplot_kw={'projection': 'polar'})

# Add black lines/arrows for a subset of phase lags
for idx in np.arange(0, len(lags), 20):
    val = lags[idx]
    phi = np.angle(val)
    r = np.abs(val)
    # axs7.arrow(phi, 0., 0., r, )
    axs7.vlines(phi, 0., r, colors='k', alpha=0.2, zorder=3)

# Red arrow for collective phase
arrow(PLV[1], 0., 0., PLV[0], alpha = 1., width = 0.015,
     edgecolor = 'red', facecolor = 'red', lw = 1, zorder = 4)

axs7.set_ylim([0,1])
axs7.set_title(r'PLV $\theta$ [{tl}-{th}]Hz - $\gamma$ [{gl}-{gh}]'.format(tl=fc_theta[0], th=fc_theta[1], gl=fc_gamma_low[0], gh=fc_gamma_low[1]))


# Polar plot #2 : theta vs high-gamma
lags, PLV = calc_PLV(EC_exc_rate_filt_theta, EC_exc_rate_filt_gamma_high)
fig8, axs8 = plt.subplots(1, figsize=(12,12), subplot_kw={'projection': 'polar'})

for idx in np.arange(0, len(lags), 20):
    val = lags[idx]
    phi = np.angle(val)
    r = np.abs(val)
    # axs7.arrow(phi, 0., 0., r, )
    axs8.vlines(phi, 0., r, colors='k', alpha=0.2, zorder=3)

# Red arrow for collective phase
arrow(PLV[1], 0., 0., PLV[0], alpha = 1., width = 0.015,
     edgecolor = 'red', facecolor = 'red', lw = 1, zorder = 4)

axs8.set_ylim([0,1])
axs8.set_title(r'PLV $\theta$ [{tl}-{th}]Hz - $\gamma$ [{gl}-{gh}]'.format(tl=fc_theta[0], th=fc_theta[1], gl=fc_gamma_high[0], gh=fc_gamma_high[1]))


plt.show()
