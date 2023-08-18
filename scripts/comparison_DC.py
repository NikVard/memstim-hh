#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

# main program
if __name__ == "__main__":

	# Define paths to compare
	datapath = "/home/nikos/Documents/projects/Python/memstim-hh/results/test_extra_noise_2mV_K_0.22nA_kN_55_stim/None/25-07-2023 12H10M26S/data"
	datapath_DC = "/home/nikos/Documents/projects/Python/memstim-hh/results/test_DC_extra_noise_2mV_K_0.22nA_kN_55/None/25-07-2023 15H24M48S/data"
	# datapath_DC = "/home/nikos/Documents/projects/Python/memstim-hh/results/test_DC_intra_inter_CA1_EC_extra_noise_2mV_K_0.22nA_kN_55/None/25-07-2023 11H38M12S/data"
	spikepath = datapath + "/spikes"
	spikepath_DC = datapath_DC + "/spikes"

	# Lower noise
	datapath = "/home/nikos/Documents/projects/Python/memstim-hh/results/test_K_0.22nA_kN_55/None/25-07-2023 16H45M57S/data"
	# no intra
	# datapath_DC = "/home/nikos/Documents/projects/Python/memstim-hh/results/test_DC_K_0.22nA_kN_55/None/25-07-2023 16H31M49S/data"
	# no intra and inter CA1->EC
	datapath_DC = "/home/nikos/Documents/projects/Python/memstim-hh/results/test_DC_intra_inter_CA1_EC_K_0.22nA_kN_55/None/25-07-2023 17H01M44S/data"
	
	# spikes
	spikepath = datapath + "/spikes"
	spikepath_DC = datapath_DC + "/spikes"

	# Outputs
	figpath = "/home/nikos/Documents/projects/Python/memstim-hh/figures/R_vs_DC_intra_inter_CA1_EC_low_noise"

	# Filenames
	filenames = ["CA1_inh_spikemon", "CA1_pyCAN_spikemon"]
	
	# Load the files
	CA1_E_i = np.loadtxt(spikepath + '/' + filenames[1] + "_i.txt")
	CA1_E_t = np.loadtxt(spikepath + '/' + filenames[1] + "_t.txt")
	CA1_I_i = np.loadtxt(spikepath + '/' + filenames[0] + "_i.txt")
	CA1_I_t = np.loadtxt(spikepath + '/' + filenames[0] + "_t.txt")
	CA1_E_i_DC = np.loadtxt(spikepath_DC + '/' + filenames[1] + "_i.txt")
	CA1_E_t_DC = np.loadtxt(spikepath_DC + '/' + filenames[1] + "_t.txt")
	CA1_I_i_DC = np.loadtxt(spikepath_DC + '/' + filenames[0] + "_i.txt")
	CA1_I_t_DC = np.loadtxt(spikepath_DC + '/' + filenames[0] + "_t.txt")

	# Parameters
	second = 1
	ms = 1e-3
	dt = 0.1*ms
	fs = 1/dt
	NE = 10000
	NI = 1000
	winsize = 1*ms
	winsamples = int(winsize*fs)
	duration = 5*second
	bin_num = int(duration/winsize)
	fs_FR = 1/winsize
	overlap = 0.9
	winsize_specg = 150*ms
	nps_specg = int(winsize_specg*fs_FR)
	nover_specg = int(overlap*nps_specg)
	winsize_PSD = 2*second
	nps_PSD = int(winsize_PSD*fs_FR)
	nover_PSD = int(overlap*nps_PSD)
	nfft_PSD = 2**(nps_PSD-1).bit_length()+2

	# FRs
	FR_E, bins_E = np.histogram(CA1_E_t, bins=bin_num, range=(0,int(duration/ms)))
	FR_I, bins_I = np.histogram(CA1_I_t, bins=bin_num, range=(0,int(duration/ms)))
	FR_E_DC, bins_E_DC = np.histogram(CA1_E_t_DC, bins=bin_num, range=(0,int(duration/ms)))
	FR_I_DC, bins_I_DC = np.histogram(CA1_I_t_DC, bins=bin_num, range=(0,int(duration/ms)))
	
	# Fix the x-axis
	bins_E_adj = bins_E[:-1] + winsize/2
	bins_I_adj = bins_I[:-1] + winsize/2
	bins_E_DC_adj = bins_E_DC[:-1] + winsize/2
	bins_I_DC_adj = bins_I_DC[:-1] + winsize/2

	# Normalize the FRs between [0,1]
	FR_E_norm = FR_E / NE
	FR_I_norm = FR_I / NI
	FR_E_norm_DC = FR_E_DC / NE
	FR_I_norm_DC = FR_I_DC /= NI

	# Spectrograms (scipy)
	fv, tv, Pxx_E = sig.spectrogram(FR_E_norm, fs=fs_FR,
										window=sig.windows.hann(M=nps_specg, sym=False),
										noverlap=nover_specg)
	_, _, Pxx_I = sig.spectrogram(FR_I_norm, fs=fs_FR,
										window=sig.windows.hann(M=nps_specg, sym=False),
										noverlap=nover_specg)
	_, _, Pxx_E_DC = sig.spectrogram(FR_E_norm_DC, fs=fs_FR,
										window=sig.windows.hann(M=nps_specg, sym=False),
										noverlap=nover_specg)
	_, _, Pxx_I_DC = sig.spectrogram(FR_I_norm_DC, fs=fs_FR,
										window=sig.windows.hann(M=nps_specg, sym=False),
										noverlap=nover_specg)

	# Colormap normalization
	vmin = 1e-12
	vmax = 10e3
	norm = colors.Normalize(vmin=vmin, vmax=vmax)
	normlog = colors.LogNorm(vmin=vmin, vmax=vmax)


	# PSDs
	## Welch function
	fv_PSD, PSD_E = sig.welch(FR_E_norm, fs_FR,
								window='hann',
								nperseg=nps_PSD,
								noverlap=nover_PSD,
								nfft=nfft_PSD,
								scaling='density',
								return_onesided=True,
								detrend='constant',
								average='mean')
	_, PSD_I = sig.welch(FR_I_norm, fs_FR,
								window='hann',
								nperseg=nps_PSD,
								noverlap=nover_PSD,
								nfft=nfft_PSD,
								scaling='density',
								return_onesided=True,
								detrend='constant',
								average='mean')
	_, PSD_E_DC = sig.welch(FR_E_norm_DC, fs_FR,
								window='hann',
								nperseg=nps_PSD,
								noverlap=nover_PSD,
								nfft=nfft_PSD,
								scaling='density',
								return_onesided=True,
								detrend='constant',
								average='mean')
	_, PSD_I_DC = sig.welch(FR_I_norm_DC, fs_FR,
								window='hann',
								nperseg=nps_PSD,
								noverlap=nover_PSD,
								nfft=nfft_PSD,
								scaling='density',
								return_onesided=True,
								detrend='constant',
								average='mean')


	# Plotting starts here
	# ----------------------------------------------------------
	# Plot 1: rasters comparison
	fig1, axs1 = plt.subplots(1, 2, figsize=(16,10), sharex=True)
	axs1[0].plot(CA1_E_t, CA1_E_i, 'b.', markersize=0.5)
	axs1[0].plot(CA1_I_t, CA1_I_i+NE, 'r.', markersize=0.5)
	axs1[0].set_title('Connected')
	axs1[0].set_ylabel('Neuron idx')
	axs1[0].set_xlabel('Time [ms]')
	axs1[0].set_ylim([0,NE+NI])

	axs1[1].plot(CA1_E_t_DC, CA1_E_i_DC, 'b.', markersize=0.5)
	axs1[1].plot(CA1_I_t_DC, CA1_I_i_DC+NE, 'r.', markersize=0.5)
	axs1[1].set_title('Disconnected')
	axs1[1].set_ylabel('Neuron idx')
	axs1[1].set_xlabel('Time [ms]')
	axs1[1].set_ylim([0,NE+NI])
	fig1.set_tight_layout(True)


	# Plot 2: FRs comparison
	fig2, axs2 = plt.subplots(2, 2, figsize=(16,10), sharey=True, sharex=True)
	axs2[0,0].plot(bins_E_adj, FR_E, 'b')
	axs2[1,0].plot(bins_I_adj, FR_I, 'r')
	axs2[0,0].set_title('Connected')
	axs2[0,0].set_ylabel('Spikerates [Hz]')
	axs2[1,0].set_ylabel('Spikerates [Hz]')
	axs2[1,0].set_xlabel('Time [ms]')

	axs2[0,1].plot(bins_E_DC_adj, FR_E_DC/NE, 'b')
	axs2[1,1].plot(bins_I_DC_adj, FR_I_DC/NI, 'r')
	axs2[0,1].set_title('Disconnected')
	axs2[1,1].set_xlabel('Time [ms]')
	axs2[1,1].set_ylim([0,1])
	fig2.set_tight_layout(True)
	

	# Plot 3: Spectrograms
	fig3, axs3 = plt.subplots(2, 2, figsize=(16,10), sharey=True, sharex=True)
	pos1 = axs3[0,0].pcolormesh(tv, fv, Pxx_E, norm=norm, shading='gouraud')
	pos2 = axs3[1,0].pcolormesh(tv, fv, Pxx_I, norm=norm, shading='gouraud')
	axs3[0,0].set_title(f'Connected\nPyCAN')
	axs3[1,0].set_title('Inhibitory')
	axs3[0,0].set_ylabel('Frequency [Hz]')
	axs3[1,0].set_ylabel('Frequency [Hz]')
	axs3[1,0].set_xlabel('Time [ms]')	
	
	pos3 = axs3[0,1].pcolormesh(tv, fv, Pxx_E_DC, norm=norm, shading='gouraud')
	pos4 = axs3[1,1].pcolormesh(tv, fv, Pxx_I_DC, norm=norm, shading='gouraud')
	axs3[0,1].set_title(f'Disconnected\nPyCAN')
	axs3[1,1].set_title('Inhibitory')
	axs3[1,1].set_xlabel('Time [ms]')
	axs3[1,1].set_ylim([0,200	])

	fig3.colorbar(pos1, ax=axs3[0,0])
	fig3.colorbar(pos2, ax=axs3[1,0])
	fig3.colorbar(pos3, ax=axs3[0,1])
	fig3.colorbar(pos4, ax=axs3[1,1])
	fig3.set_tight_layout(True)

	# Plot 4: PSDs
	fig4, axs4 = plt.subplots(1, 2, figsize=(16,10), sharey=True, sharex=True)
	axs4[0].plot(fv_PSD, PSD_E, 'b')
	axs4[0].plot(fv_PSD, PSD_I, 'r')
	axs4[0].set_title('Connected')
	axs4[0].set_ylabel('PSD')
	axs4[0].set_xlabel('Frequency [Hz]')	

	axs4[1].plot(fv_PSD, PSD_E_DC, 'b')
	axs4[1].plot(fv_PSD, PSD_I_DC, 'r')
	axs4[1].set_title('Disconnected')
	axs4[1].set_xlabel('Frequency [Hz]')
	axs4[1].set_xlim([0,200])
	fig4.set_tight_layout(True)

	# Save the figs
	fig1.savefig(figpath + "/rasters.png")
	fig2.savefig(figpath + "/FRs.png")
	fig3.savefig(figpath + "/specgrams.png")
	fig4.savefig(figpath + "/PSDs.png")

	# Show
	plt.show()