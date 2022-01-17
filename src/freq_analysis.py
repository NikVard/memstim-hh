from scipy import signal

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
