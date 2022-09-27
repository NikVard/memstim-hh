import numpy as np
from scipy import signal as sig

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
                                # detrend=False,
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
