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

figure()
tl = tiledlayout(4,1,'TileSpacing','Compact');

% Tile 1
ax(1) = nexttile;
plot(fv, EC.exc.PSD, 'DisplayName','Excitatory [Py]')
hold on
plot(fv, EC.inh.PSD, 'DisplayName','Inhibitory [Inh]')
title('EC')

% Tile 2
ax(2) = nexttile;
plot(fv, DG.exc.PSD)
hold on
plot(fv, DG.inh.PSD)
title('DG')

% Tile 3
ax(3) = nexttile;
plot(fv, CA3.exc.PSD)
hold on
plot(fv, CA3.inh.PSD)
title('CA3')

% Tile 4
ax(4) = nexttile;
plot(fv, CA1.exc.PSD)
hold on
plot(fv, CA1.inh.PSD)
title('CA1')

xlabel('Frequency [Hz]')
linkaxes(ax,'x')

lh = legend(ax(1),'Location','NorthOutside','Orientation','Horizontal');
lh.Layout.Tile = 'North'; % <----- relative to tiledlayout


%% Time-Frequency Analysis (Spectrogram w/ STFT)
% -----------------------------------------------
t_res = 0.25;

figure()
tl = tiledlayout(4,2,'TileSpacing','Compact');

% Tile 1/1
ax(1,1) = nexttile;
pspectrum(EC.exc.rates_dm, fs, 'spectrogram', 'OverlapPercent', 99, 'TimeResolution', t_res)

% Tile 1/2
ax(1,2) = nexttile;
pspectrum(EC.inh.rates_dm, fs, 'spectrogram', 'OverlapPercent', 99, 'TimeResolution', t_res)


% Tile 2/1
ax(2,1) = nexttile;
pspectrum(DG.exc.rates_dm, fs, 'spectrogram', 'OverlapPercent', 99, 'TimeResolution', t_res)

% Tile 2/2
ax(2,2) = nexttile;
pspectrum(DG.inh.rates_dm, fs, 'spectrogram', 'OverlapPercent', 99, 'TimeResolution', t_res)


% Tile 3/1
ax(3,1) = nexttile;
pspectrum(CA3.exc.rates_dm, fs, 'spectrogram', 'OverlapPercent', 99, 'TimeResolution', t_res)

% Tile 3/2
ax(3,2) = nexttile;
pspectrum(CA3.inh.rates_dm, fs, 'spectrogram', 'OverlapPercent', 99, 'TimeResolution', t_res)


% Tile 4/1
ax(4,1) = nexttile;
pspectrum(CA1.exc.rates_dm, fs, 'spectrogram', 'OverlapPercent', 99, 'TimeResolution', t_res)

% Tile 4/2
ax(4,2) = nexttile;
pspectrum(CA1.inh.rates_dm, fs, 'spectrogram', 'OverlapPercent', 99, 'TimeResolution', t_res)


%% Time-Frequency Analysis (Chronux)
% -----------------------------------
params = {};
params.Fs = fs;                     % sampling frequency
params.fpass = [0 fs/2];            % frequencies in output
params.pad = 1;                     % -1: no padding; 0: next power of 2; 1: next power of 2+1
params.error = 0;                   % error bar calculation; 0: none; 1: theoretical; 2: jackknife
params.trialave = 0;                % trial averaging; not required
params.tapers = [5,9];              % 
params.movingwin = [0.25, 0.01];    % 2-elem array; [size of the moving window, step size to advance the window]


[EC.exc.TFR.S,EC.exc.TFR.t,EC.exc.TFR.f] = mtspecgramc( EC.exc.rates_dm, params.movingwin, params );
[EC.inh.TFR.S,EC.inh.TFR.t,EC.inh.TFR.f] = mtspecgramc( EC.inh.rates_dm, params.movingwin, params );

[DG.exc.TFR.S,DG.exc.TFR.t,DG.exc.TFR.f] = mtspecgramc( DG.exc.rates_dm, params.movingwin, params );
[DG.inh.TFR.S,DG.inh.TFR.t,DG.inh.TFR.f] = mtspecgramc( DG.inh.rates_dm, params.movingwin, params );

[CA3.exc.TFR.S,CA3.exc.TFR.t,CA3.exc.TFR.f] = mtspecgramc( CA3.exc.rates_dm, params.movingwin, params );
[CA3.inh.TFR.S,CA3.inh.TFR.t,CA3.inh.TFR.f] = mtspecgramc( CA3.inh.rates_dm, params.movingwin, params );

[CA1.exc.TFR.S,CA1.exc.TFR.t,CA1.exc.TFR.f] = mtspecgramc( CA1.exc.rates_dm, params.movingwin, params );
[CA1.inh.TFR.S,CA1.inh.TFR.t,CA1.inh.TFR.f] = mtspecgramc( CA1.inh.rates_dm, params.movingwin, params );

figure()

subplot(4,2,1)
axs(1,1) = imagesc('XData', EC.exc.TFR.t.', 'YData', EC.exc.TFR.f, 'CData', EC.exc.TFR.S.');
xlim([EC.exc.TFR.t(1), EC.exc.TFR.t(end)]);
ylim(params.fpass);

subplot(4,2,2)
axs(1,2) = imagesc('XData', EC.inh.TFR.t.', 'YData', EC.inh.TFR.f, 'CData', EC.inh.TFR.S.');
xlim([EC.inh.TFR.t(1), EC.inh.TFR.t(end)]);
ylim(params.fpass);

subplot(4,2,3)
axs(2,1) = imagesc('XData', DG.exc.TFR.t.', 'YData', DG.exc.TFR.f, 'CData', DG.exc.TFR.S.');
xlim([DG.exc.TFR.t(1), DG.exc.TFR.t(end)]);
ylim(params.fpass);

subplot(4,2,4)
axs(2,2) = imagesc('XData', DG.inh.TFR.t.', 'YData', DG.inh.TFR.f, 'CData', DG.inh.TFR.S.');
xlim([DG.inh.TFR.t(1), DG.inh.TFR.t(end)]);
ylim(params.fpass);

subplot(4,2,5)
axs(3,1) = imagesc('XData', CA3.exc.TFR.t.', 'YData', CA3.exc.TFR.f, 'CData', CA3.exc.TFR.S.');
xlim([CA3.exc.TFR.t(1), CA3.exc.TFR.t(end)]);
ylim(params.fpass);

subplot(4,2,6)
axs(3,2) = imagesc('XData', CA3.inh.TFR.t.', 'YData', CA3.inh.TFR.f, 'CData', CA3.inh.TFR.S.');
xlim([CA3.inh.TFR.t(1), CA3.inh.TFR.t(end)]);
ylim(params.fpass);

subplot(4,2,7)
axs(4,1) = imagesc('XData', CA1.exc.TFR.t.', 'YData', CA1.exc.TFR.f, 'CData', CA1.exc.TFR.S.');
xlim([CA1.exc.TFR.t(1), CA1.exc.TFR.t(end)]);
ylim(params.fpass);

subplot(4,2,8)
axs(4,2) = imagesc('XData', CA1.inh.TFR.t.', 'YData', CA1.inh.TFR.f, 'CData', CA1.inh.TFR.S.');
xlim([CA1.inh.TFR.t(1), CA1.inh.TFR.t(end)]);
ylim(params.fpass);


%% Filtering and Hilbert Transform
% ---------------------------------
% Defining cutoff frequencies
fc = {};
fc.low = [12];
fc.theta = [4, 12];
fc.gammal = [30, 80];
fc.gammah = [80, 200]; % Canolty/Knight 2010
fc.high = [200];

% Filter the data
% EC
data = EC.exc.rates_dm;
EC.exc.filt.low = filter_pass(data, fs, [], fc.low(1), 8);
EC.exc.filt.theta = filter_pass(data, fs, fc.theta(1), fc.theta(2), 8);
EC.exc.filt.gammal = filter_pass(data, fs, fc.gammal(1), fc.gammal(2), 8);
EC.exc.filt.gammah = filter_pass(data, fs, fc.gammah(1), fc.gammah(2), 8);

data = EC.inh.rates_dm;
EC.inh.filt.low = filter_pass(data, fs, [], fc.low(1), 8);
EC.inh.filt.theta = filter_pass(data, fs, fc.theta(1), fc.theta(2), 8);
EC.inh.filt.gammal = filter_pass(data, fs, fc.gammal(1), fc.gammal(2), 8);
EC.inh.filt.gammah = filter_pass(data, fs, fc.gammah(1), fc.gammah(2), 8);

% DG
data = DG.exc.rates_dm;
DG.exc.filt.low = filter_pass(data, fs, [], fc.low(1), 8);
DG.exc.filt.theta = filter_pass(data, fs, fc.theta(1), fc.theta(2), 8);
DG.exc.filt.gammal = filter_pass(data, fs, fc.gammal(1), fc.gammal(2), 8);
DG.exc.filt.gammah = filter_pass(data, fs, fc.gammah(1), fc.gammah(2), 8);

data = DG.inh.rates_dm;
DG.inh.filt.low = filter_pass(data, fs, [], fc.low(1), 8);
DG.inh.filt.theta = filter_pass(data, fs, fc.theta(1), fc.theta(2), 8);
DG.inh.filt.gammal = filter_pass(data, fs, fc.gammal(1), fc.gammal(2), 8);
DG.inh.filt.gammah = filter_pass(data, fs, fc.gammah(1), fc.gammah(2), 8);

% CA3
data = CA3.exc.rates_dm;
CA3.exc.filt.low = filter_pass(data, fs, [], fc.low(1), 8);
CA3.exc.filt.theta = filter_pass(data, fs, fc.theta(1), fc.theta(2), 8);
CA3.exc.filt.gammal = filter_pass(data, fs, fc.gammal(1), fc.gammal(2), 8);
CA3.exc.filt.gammah = filter_pass(data, fs, fc.gammah(1), fc.gammah(2), 8);

data = CA3.inh.rates_dm;
CA3.inh.filt.low = filter_pass(data, fs, [], fc.low(1), 8);
CA3.inh.filt.theta = filter_pass(data, fs, fc.theta(1), fc.theta(2), 8);
CA3.inh.filt.gammal = filter_pass(data, fs, fc.gammal(1), fc.gammal(2), 8);
CA3.inh.filt.gammah = filter_pass(data, fs, fc.gammah(1), fc.gammah(2), 8);

% CA1
data = CA1.exc.rates_dm;
CA1.exc.filt.low = filter_pass(data, fs, [], fc.low(1), 8);
CA1.exc.filt.theta = filter_pass(data, fs, fc.theta(1), fc.theta(2), 8);
CA1.exc.filt.gammal = filter_pass(data, fs, fc.gammal(1), fc.gammal(2), 8);
CA1.exc.filt.gammah = filter_pass(data, fs, fc.gammah(1), fc.gammah(2), 8);

data = CA1.inh.rates_dm;
CA1.inh.filt.low = filter_pass(data, fs, [], fc.low(1), 8);
CA1.inh.filt.theta = filter_pass(data, fs, fc.theta(1), fc.theta(2), 8);
CA1.inh.filt.gammal = filter_pass(data, fs, fc.gammal(1), fc.gammal(2), 8);
CA1.inh.filt.gammah = filter_pass(data, fs, fc.gammah(1), fc.gammah(2), 8);

figure()
plot(tv(1:end-1), EC.exc.rates_dm)
hold on;
plot(tv(1:end-1), EC.exc.filt.low)
plot(tv(1:end-1), EC.exc.filt.theta)
plot(tv(1:end-1), EC.exc.filt.gammal)
plot(tv(1:end-1), EC.exc.filt.gammah)
legend('demean', 'low', 'theta', 'gamma low', 'gamma high')

% Hilbert transform to extract phase/amplitude
data_ht = hilbert(data);


%% CFC and PAC between Theta and Gamma rhythms
% ---------------------------------------------
% figure()
% plot(tv(1:end-1), EC.exc.filt.theta)
% hold on
% plot(tv(1:end-1), EC.exc.filt.gammah)

% PLV calculation
EC.exc.analytic.theta = hilbert(EC.exc.filt.theta);
EC.exc.analytic.gammal = hilbert(EC.exc.filt.gammal);
EC.exc.analytic.gammah= hilbert(EC.exc.filt.gammah);
EC.exc.analytic.gammal2 = hilbert(real(EC.exc.analytic.gammal));
EC.exc.analytic.gammah2 = hilbert(real(EC.exc.analytic.gammah));

% Calculate the phase differences
phi_theta = angle(EC.exc.analytic.theta);
phi_gammah = angle(EC.exc.analytic.gammah2);
phi_d = phi_theta - phi_gammah;
lags = exp(1j*phi_d);

% Calculate the PLV (mean of lags)
mu_phi = mean(lags);
PLV = [abs(mu_phi), angle(mu_phi)];

figure()
polarplot([angle(lags); angle(lags)], [zeros(1,length(lags)); ones(1,length(lags))], 'Color', [0,0,0,0.1])
hold on
polarplot([PLV(2); PLV(2)], [0; PLV(1)], 'Color', 'red')

if PLV(1) < 0.1
    disp('Nothing there...')
end


%% Coherency between traces of theta and gamma
% ---------------------------------------------
params = {};
params.Fs = fs;                     % sampling frequency
params.fpass = [0 250];             % frequencies in output
params.pad = 1;                     % -1: no padding; 0: next power of 2; 1: next power of 2+1
params.error = 0;                   % error bar calculation; 0: none; 1: theoretical; 2: jackknife
params.trialave = 0;                % trial averaging; not required
params.tapers = [3,5];              % 
params.movingwin = [0.25, 0.01];    % 2-elem array; [size of the moving window, step size to advance the window]

[EC.exc.coh.C,EC.exc.coh.phi, EC.exc.coh.S12,EC.exc.coh.S1,EC.exc.coh.S2,EC.exc.coh.t, EC.exc.coh.f] = cohgramc(EC.exc.filt.gammal.',EC.exc.rates_dm.',params.movingwin,params);
[EC.inh.coh.C,EC.inh.coh.phi, EC.inh.coh.S12,EC.inh.coh.S1,EC.inh.coh.S2,EC.inh.coh.t, EC.inh.coh.f] = cohgramc(EC.inh.filt.gammal.',EC.inh.rates_dm.',params.movingwin,params);
[DG.exc.coh.C,DG.exc.coh.phi, DG.exc.coh.S12,DG.exc.coh.S1,DG.exc.coh.S2,DG.exc.coh.t, DG.exc.coh.f] = cohgramc(DG.exc.filt.gammal.',DG.exc.rates_dm.',params.movingwin,params);
[DG.inh.coh.C,DG.inh.coh.phi, DG.inh.coh.S12,DG.inh.coh.S1,DG.inh.coh.S2,DG.inh.coh.t, DG.inh.coh.f] = cohgramc(DG.inh.filt.gammal.',DG.inh.rates_dm.',params.movingwin,params);
[CA3.exc.coh.C,CA3.exc.coh.phi, CA3.exc.coh.S12,CA3.exc.coh.S1,CA3.exc.coh.S2,CA3.exc.coh.t, CA3.exc.coh.f] = cohgramc(CA3.exc.filt.gammal.',CA3.exc.rates_dm.',params.movingwin,params);
[CA3.inh.coh.C,CA3.inh.coh.phi, CA3.inh.coh.S12,CA3.inh.coh.S1,CA3.inh.coh.S2,CA3.inh.coh.t, CA3.inh.coh.f] = cohgramc(CA3.inh.filt.gammal.',CA3.inh.rates_dm.',params.movingwin,params);
[CA1.exc.coh.C,CA1.exc.coh.phi, CA1.exc.coh.S12,CA1.exc.coh.S1,CA1.exc.coh.S2,CA1.exc.coh.t, CA1.exc.coh.f] = cohgramc(CA1.exc.filt.gammal.',CA1.exc.rates_dm.',params.movingwin,params);
[CA1.inh.coh.C,CA1.inh.coh.phi, CA1.inh.coh.S12,CA1.inh.coh.S1,CA1.inh.coh.S2,CA1.inh.coh.t, CA1.inh.coh.f] = cohgramc(CA1.inh.filt.gammal.',CA1.inh.rates_dm.',params.movingwin,params);

figure()
subplot(4,2,1)
axs(1,1) = imagesc('XData', EC.exc.coh.t.', 'YData', EC.exc.coh.f, 'CData', EC.exc.coh.C.');
xlim([EC.exc.coh.t(1), EC.exc.coh.t(end)]);
ylim(params.fpass);

subplot(4,2,2)
axs(1,2) = imagesc('XData', EC.inh.coh.t.', 'YData', EC.inh.coh.f, 'CData', EC.inh.coh.C.');
xlim([EC.inh.coh.t(1), EC.inh.coh.t(end)]);
ylim(params.fpass);

subplot(4,2,3)
axs(2,1) = imagesc('XData', DG.exc.coh.t.', 'YData', DG.exc.coh.f, 'CData', DG.exc.coh.C.');
xlim([DG.exc.coh.t(1), DG.exc.coh.t(end)]);
ylim(params.fpass);

subplot(4,2,4)
axs(2,2) = imagesc('XData', DG.inh.coh.t.', 'YData', DG.inh.coh.f, 'CData', DG.inh.coh.C.');
xlim([DG.inh.coh.t(1), DG.inh.coh.t(end)]);
ylim(params.fpass);

subplot(4,2,5)
axs(3,1) = imagesc('XData', CA3.exc.coh.t.', 'YData', CA3.exc.coh.f, 'CData', CA3.exc.coh.C.');
xlim([CA3.exc.coh.t(1), CA3.exc.coh.t(end)]);
ylim(params.fpass);

subplot(4,2,6)
axs(3,2) = imagesc('XData', CA3.inh.coh.t.', 'YData', CA3.inh.coh.f, 'CData', CA3.inh.coh.C.');
xlim([CA3.inh.coh.t(1), CA3.inh.coh.t(end)]);
ylim(params.fpass);

subplot(4,2,7)
axs(4,1) = imagesc('XData', CA1.exc.coh.t.', 'YData', CA1.exc.coh.f, 'CData', CA1.exc.coh.C.');
xlim([CA1.exc.coh.t(1), CA1.exc.coh.t(end)]);
ylim(params.fpass);

subplot(4,2,8)
axs(4,2) = imagesc('XData', CA1.inh.coh.t.', 'YData', CA1.inh.coh.f, 'CData', CA1.inh.coh.C.');
xlim([CA1.inh.coh.t(1), CA1.inh.coh.t(end)]);
ylim(params.fpass);


%% Cross-correlation between traces of theta and gamma
% -----------------------------------------------------
EC.exc.xcorr = xcorr(EC.exc.filt.theta, EC.exc.filt.gammal, [], 'coeff'); % normalized to 1

figure()
plot([-tv(end-2:-1:1), 0, tv(1:end-2)], EC.exc.xcorr)
