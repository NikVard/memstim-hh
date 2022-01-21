close all
clear
clc

%% Parameters
% ------------
T = 2; %  duration, seconds
dt = 1e-3; % 1ms
fs = 1/dt; % 1kHz
bin_size = 1e-3; % 1ms

areas = {'EC', 'DG', 'CA3', 'CA1'};
types = {'pyCAN', 'py', 'inh'};

dirs = {};
dirs.results = '../../results/';
dirs.data = strcat(dirs.results, 'analysis/data/');
dirs.spikes = strcat(dirs.data, 'spikes/');

EC.exc.type = types{1};
EC.exc.N = 10000;
EC.inh.type = types{3};
EC.inh.N = 1000;
DG.exc.type = types{2};
DG.exc.N = 10000;
DG.inh.type = types{3};
DG.inh.N = 100;
CA3.exc.type = types{1};
CA3.exc.N = 1000;
CA3.inh.type = types{3};
CA3.inh.N = 100;
CA1.exc.type = types{1};
CA1.exc.N = 10000;
CA1.inh.type = types{3};
CA1.inh.N = 1000;

%% Read the spike data
% ---------------------
% EC
fname = 'EC_pyCAN_spikemon_t.txt';
EC.exc.t = file_read(strcat(dirs.spikes, fname), '%f');
fname = 'EC_inh_spikemon_t.txt';
EC.inh.t = file_read(strcat(dirs.spikes, fname), '%f');

% DG
fname = 'DG_py_spikemon_t.txt';
DG.exc.t = file_read(strcat(dirs.spikes, fname), '%f');
fname = 'DG_inh_spikemon_t.txt';
DG.inh.t = file_read(strcat(dirs.spikes, fname), '%f');

% CA3
fname = 'CA3_pyCAN_spikemon_t.txt';
CA3.exc.t = file_read(strcat(dirs.spikes, fname), '%f');
fname = 'CA3_inh_spikemon_t.txt';
CA3.inh.t = file_read(strcat(dirs.spikes, fname), '%f');

% CA1
fname = 'CA1_pyCAN_spikemon_t.txt';
CA1.exc.t = file_read(strcat(dirs.spikes, fname), '%f');
fname = 'CA1_inh_spikemon_t.txt';
CA1.inh.t = file_read(strcat(dirs.spikes, fname), '%f');


%% From spikes to rates (faux-LFP)
% ---------------------------------
bin_width = bin_size/dt;
[EC.exc.rates, tv] = histcounts(EC.exc.t, T/bin_size, 'Normalization', 'count');
[EC.inh.rates, ~] = histcounts(EC.inh.t, T/bin_size, 'Normalization', 'count');
[DG.exc.rates, ~] = histcounts(DG.exc.t, T/bin_size, 'Normalization', 'count');
[DG.inh.rates, ~] = histcounts(DG.exc.t, T/bin_size, 'Normalization', 'count');
[CA3.exc.rates, ~] = histcounts(CA3.exc.t, T/bin_size, 'Normalization', 'count');
[CA3.inh.rates, ~] = histcounts(CA3.exc.t, T/bin_size, 'Normalization', 'count');
[CA1.exc.rates, ~] = histcounts(CA1.exc.t, T/bin_size, 'Normalization', 'count');
[CA1.inh.rates, ~] = histcounts(CA1.exc.t, T/bin_size, 'Normalization', 'count');

% Normalize from (spikes per bin) to (spikes per area per second)
EC.exc.rates = EC.exc.rates/EC.exc.N/bin_size;
EC.inh.rates = EC.inh.rates/EC.inh.N/bin_size;
DG.exc.rates = DG.exc.rates/DG.exc.N/bin_size;
DG.inh.rates = DG.inh.rates/DG.inh.N/bin_size;
CA3.exc.rates = CA3.exc.rates/CA3.exc.N/bin_size;
CA3.inh.rates = CA3.inh.rates/CA3.inh.N/bin_size;
CA1.exc.rates = CA1.exc.rates/CA1.exc.N/bin_size;
CA1.inh.rates = CA1.inh.rates/CA1.inh.N/bin_size;

% Demean the data
EC.exc.rates_dm = detrend(EC.exc.rates);
EC.inh.rates_dm = detrend(EC.inh.rates);
DG.exc.rates_dm = detrend(DG.exc.rates);
DG.inh.rates_dm = detrend(DG.inh.rates);
CA3.exc.rates_dm = detrend(CA3.exc.rates);
CA3.inh.rates_dm = detrend(CA3.inh.rates);
CA1.exc.rates_dm = detrend(CA1.exc.rates);
CA1.inh.rates_dm = detrend(CA1.inh.rates);

% plot(tv(1:end-1), (EC.exc.rates/EC.exc.N)/bin_size)
% hold on
% plot(tv(1:end-1), DG.exc.rates/DG.exc.N/bin_size)

%% Power Spectral Density (Single-Sided periodogram)
% ---------------------------------------------------
NFFT = 2^nextpow2(length(tv));

[EC.exc.PSD, fv] = calc_PSD(EC.exc.rates_dm, fs, 'NFFT', NFFT);
[EC.inh.PSD, ~] = calc_PSD(EC.inh.rates_dm, fs, 'NFFT', NFFT);
[DG.exc.PSD, ~] = calc_PSD(DG.exc.rates_dm, fs, 'NFFT', NFFT);
[DG.inh.PSD, ~] = calc_PSD(DG.inh.rates_dm, fs, 'NFFT', NFFT);
[CA3.exc.PSD, ~] = calc_PSD(CA3.exc.rates_dm, fs, 'NFFT', NFFT);
[CA3.inh.PSD, ~] = calc_PSD(CA3.inh.rates_dm, fs, 'NFFT', NFFT);
[CA1.exc.PSD, ~] = calc_PSD(CA1.exc.rates_dm, fs, 'NFFT', NFFT);
[CA1.inh.PSD, ~] = calc_PSD(CA1.inh.rates_dm, fs, 'NFFT', NFFT);

plot(fv, EC.exc.PSD)


