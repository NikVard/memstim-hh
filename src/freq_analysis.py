import numpy as np
from scipy import signal as sig

# Filtering functions
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

# Functions examples from https://www.programcreek.com/python/example/100546/scipy.signal.spectrogram
def power_spectrum(signal: np.ndarray,
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
    f, t, sxx = spectrogram(x=signal,
                            fs=fs,
                            window=hann(window_width, sym=False),
                            noverlap=window_overlap,
                            mode="magnitude")

    return f, t, (1.0 / window_width) * (sxx ** 2)



def power_to_db(spectrum: np.ndarray,
                clip_below: float = None,
                clip_above: float = None) -> np.ndarray:
    """
    Convert a spectrogram to the Decibel scale.

    Optionally, frequencies with amplitudes below or above a certain threshold can be clipped.

    Parameters
    ----------
    spectrum: numpy.ndarray
        The spectrogram to convert
    clip_below: float, optional
        Clip frequencies below the specified amplitude in dB
    clip_above: float, optional
        Clip frequencies above the specified amplitude in dB

    Returns
    -------
    numpy.ndarray
        The spectrogram on the Decibel scale
    """
    # there might be zeros, fix them to the lowest non-zero power in the spectrogram
    epsilon = np.min(spectrum[np.where(spectrum > 0)])

    sxx = np.where(spectrum > epsilon, spectrum, epsilon)
    sxx = 10 * np.log10(sxx / np.max(sxx))

    if clip_below is not None:
        sxx = np.maximum(sxx, clip_below)

    if clip_above is not None:
        sxx = np.minimum(sxx, clip_above)

    return sxx


def my_FR(spikes: np.ndarray,
            duration: int,
            window_size: float,
            overlap: float) -> (np.ndarray, np.ndarray):
    """
    Compute the firing rate using a windowed moving average.

    Parameters
    ----------
    spikes: numpy.ndarray
        The spike times (*not* Brian2 format, in -unitless- seconds)
    duration: int
        The duration of the recording (in -unitless- seconds)
    window_size: float
        Width of the moving average window (in -unitless- seconds)
    overlap: float
        Desired overlap between the windows (percentage in [0., 1.))

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
                   window_overlap: int,
                   k: int = 1,
                   **kwargs: dict) -> (np.ndarray, np.ndarray, np.ndarray):
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
    k: int
        Used in the nfft calculation; round to (nearest power of 2)+k
    kwargs: dict
        Extra arguments to pass to signal.spectrogram(); for more info, refer to the scipy documentation.

    Returns
    -------
    f: numpy.ndarray
        Array of frequency values for the first axis of the returned spectrogram
    t: numpy.ndarray
        Array of time values for the second axis of the returned spectrogram
    Sxx: numpy.ndarray
        Power spectrogram of the input signal with axes [frequency, time]
    """
    nfft = 2**((window_width-1).bit_length()+k) # np.ceil(np.log2(window_width))
    f, t, Sxx = sig.spectrogram(x=signal,
                                fs=fs,
                                nfft=nfft,
                                detrend='constant',
                                window=sig.windows.hann(M=window_width, sym=False),
                                # nperseg=window_width,
                                noverlap=window_overlap,
                                **kwargs)

    return f, t, Sxx
    # ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel


def my_PSD(signal: np.ndarray,
                   fs: int,
                   window_width: int,
                   window_overlap: int,
                   k: int = 1,
                   **kwargs: dict) -> (np.ndarray, np.ndarray):
    """
    Computes the Power Spectral Density (PSD) of the specified signal using Welch's method.

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
    k: int
        Used in the nfft calculation; round to (nearest power of 2)+k
    kwargs: dict
        Extra arguments to pass to signal.welch(); for more info, refer to the scipy documentation.

    Returns
    -------
    f: numpy.ndarray
        Array of frequency values for the first axis of the PSD
    Pxx: numpy.ndarray
        PSD of the input signal.
    """
    nfft = 2**((window_width-1).bit_length()+k) # np.ceil(np.log2(window_width))
    f, Pxx = sig.welch(x=signal,
                        fs=fs,
                        nfft=nfft,
                        window=sig.windows.hann(M=window_width, sym=False),
                        # window='boxcar',
                        nperseg=window_width,
                        noverlap=window_overlap,
                        **kwargs)

    return f, Pxx


def my_modulation_index(sig_phase: np.ndarray,
                        sig_amp: np.ndarray,
                        nbins: int=18) -> (float, float):
    """
    Computes the Modulation Index between two signals. The formalism used is the
    one provided in the following paper:

        # REF #
        Tort AB, Komorowski R, Eichenbaum H, Kopell N. Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. J Neurophysiol. 2010 Aug;104(2):1195-210. doi: 10.1152/jn.00106.2010. Epub 2010 May 12. Erratum in: J Neurophysiol. 2010 Oct;104(4):2302. PubMed PMID: 20463205; PubMed Central PMCID: PMC2941206.
        # --- #

    Modulation Index (MI) is returned as a single number; the computation is based in the Kullback-Leibler distance. More details in the paper (p. 1196-7).

    Parameters
    ----------
    sig_phase: numpy.ndarray
        The phase signal xfp(t)
    sig_amp: numpy.ndarray
        The amplitude signal xfA(t).
    nbins: int
        Number of phase bins

    Returns
    -------
    MI: float
        Modulation Index, as computed in the paper by Tort et al. (2010)
    dist_KL: float
        Kullback-Leibler distance.
    """

    # Make the bins
    bin_edges = np.linspace(-np.pi, np.pi, nbins+1)
    bin_centers = bin_edges[1:] - np.diff(bin_edges)/2

    # Allocate to bins using their indices
    idx_bin = np.digitize(sig_phase, bin_edges)
    bin_amp = np.zeros(nbins)
    for bin in np.arange(nbins):
        if np.any(idx_bin == bin):
            bin_amp[bin] = np.mean(sig_amp[idx_bin == bin])

    # Hist. normalization step - get P(j) vals
    P_amp = bin_amp / np.sum(bin_amp)

    # Kullback-Leibler distance & modulation index (MI)
    Q_amp = np.ones(nbins) / nbins

    # In the special case where observed probability in a bin is 0, this tweak
    # allows computing a meaningful KL distance nonetheless
    P_amp[np.where(P_amp == 0)] = 1e-12

    # Kullback-Leibler distance
    dist_KL = np.sum(P_amp * np.log(P_amp / Q_amp))

    # Modulation Index
    MI = dist_KL / np.log(nbins)

    return MI, dist_KL


def bandpower(data, fs, band, window_sec=None, overlap=0.9, relative=False, return_PSD=False, **kwargs):
    """
    Compute the average power of the signal x in a specific frequency band.
    Code adapted from: https://raphaelvallat.com/bandpower.html

    Parameters
    ----------
    data: 1d-array
        Input signal in the time-domain.
    fs: float
        Sampling frequency of the data.
    band: list
        Lower and upper frequencies of the band of interest.
    window_sec: float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative: boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        window_width = int(window_sec * fs)
    else:
        window_width = int((2 / low) * fs)

    # Calculate window overlap
    window_overlap = int(overlap * window_width)

    # Compute the modified periodogram (Welch)
    nfft = 2**((window_width-1).bit_length()+1)
    freqs, psd = sig.welch(x=data,
                           fs=fs,
                           nfft=nfft,
                           window=sig.windows.hann(M=window_width, sym=False),
                           # window='boxcar',
                           nperseg=window_width,
                           noverlap=window_overlap,
                           **kwargs)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    if psd.size > 1:
        psd = psd.squeeze()
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)

    if return_PSD:
        return bp, freqs, psd

    return bp
