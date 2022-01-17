from scipy import signal


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


def butter_bandpass_compare(lowcut, highcut, fs, order=8):
    ''' Comparison of [b,a] and sos filter formats. SOS (second-order sections) is better. '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos_out = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    b, a = signal.butter(order, [low, high], analog=False, btype='band', output='ba')

    return b,a,sos_out
