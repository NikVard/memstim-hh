import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.spatial.distance as dst
import csv

# Data processing functions
def my_FR(spikes: np.ndarray,
            duration: float,
            window_size: float,
            overlap: float) -> (np.ndarray, np.ndarray):
    """
    Compute the firing rate using a windowed moving average.

    Parameters
    ----------
    spikes: numpy.ndarray
        The spike times (Brian2 format, in _seconds_)
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
    fs_n = int(1/win_step)

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

    FR = (np.array(counts)/window_size)

    # return centers, spike counts, and adjusted sampling rates per window
    return centers, FR, fs_n


def my_PSD(data, fs, N):

    # Welch estimate parameters
    segment_size = np.int32(0.5*N) # Segment size = 50 % of data length
    overlap_frac = 0.5
    overlap_size = overlap_frac*segment_size
    fft_size = 512

    # Frequency resolution
    fres = fs/segment_size

    ## Welch function
    f, PSD_welch = signal.welch(data, fs, window='hann', nperseg=segment_size, noverlap=overlap_size, nfft=fft_size, scaling='density', return_onesided=True, detrend='constant', average='mean')

    return f, PSD_welch


# Optimization Functions
def logistic0(x, L=1, k=1, xm=0):
    """
    Logistic function for parameter optimization.

    Parameters
    ----------
    L: np.float32
        Maximum value (defaults to 1).
    k: np.float32
        Controls the rate of the decay.
    xm: np.float32
        Point where f(xm) = 1/2.

    Returns
    -------
    y: np.ndarray
        y = f(x) = L / (1 + exp(-k * (x-xm)))
    """

    return L / (1 + np.exp(-k * (x-xm)))


def logistic1(x, L=1, k=1, xm=0):
    """
    Logistic function (y-mirrored) for parameter optimization.

    Parameters
    ----------
    x: np.ndarray
        Input values.
    L: np.float32
        Maximum value (defaults to 1).
    k: np.float32
        Controls the rate of the decay.
    xm: np.float32
        Point where f(xm) = 1/2.

    Returns
    -------
    y: np.ndarray
        y = f(x) = L / (1 + exp(-k * (-x+xm)))
    """

    return L / (1 + np.exp(-k * (-x+xm)))


def my_tanh(x, gp=1, gn=1, uz=1, uo=0):
    """
    Hyperbolic tangent (uneven) for parameter optimization.

    Parameters
    ----------
    x: np.ndarray
        Input values.
    gp: np.float32
        Controls the gain of the _positive_ part.
    gn: np.float32
        Controls the gain of the _negative_ part.
    uz: np.float32
        Point for the zero-crossing.
    uo: np.float32
        Vertical (y-axis) offset.

    Returns
    -------
    y: np.ndarray
        y = g0*tanh(x-u0)*(x>=u0) + g1*tanh(x-u0)*(x<u0)
    """

    pos = gp*(np.tanh(x-uz)+uo) + uo
    neg = gn*(np.tanh(x-uz)+uo) + uo

    return pos*(x>=uz) + neg*(x<uz)


def my_gauss(x, mu=0, sigma=0.5, g=1):
    """
    Normalized Gaussian for parameter optimization.

    Parameters
    ----------
    x: np.ndarray
        Input values.
    sigma: np.float32
        Variance.
    mu: np.float32
        Mean.
    g: np.float32
        Output gain.

    Returns
    -------
    y: np.ndarray
        y = 1/(sigma*sqrt(2*pi)) * exp((-(x-mu)**2)/(2*sigma**2))
    """
    G = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*(sigma**2)))
    G /= np.max(G)

    return g*G


def cost_func(data, target=None, duration=1, fs=10000, areasizes={"EC":[10000, 1000], "DG":[10000, 100], "CA3":[1000, 100], "CA1":[10000, 1000]}, params_FR={"winsize":1e3, "overlap":0.5}, params_PSD={"winsize":1e3}, gains={"FR":1, "FR_mean":1, "FR_max":1, "PSD":1}):

    # Parameters
    N_EC_exc = areasizes["EC"][0]
    N_EC_inh = areasizes["EC"][1]
    N_DG_exc = areasizes["DG"][0]
    N_DG_inh = areasizes["DG"][1]
    N_CA3_exc = areasizes["CA3"][0]
    N_CA3_inh = areasizes["CA3"][1]
    N_CA1_exc = areasizes["CA1"][0]
    N_CA1_inh = areasizes["CA1"][1]

    winsize_FR = params_FR["winsize"]
    overlap_FR = params_FR["overlap"]

    winsize_PSD = params_PSD["winsize"]

    g0 = gains["FR"]
    g1 = gains["FR_mean"]
    g2 = gains["FR_max"]
    g3 = gains["PSD"]

    # Goal vector [mu_FR_E mu_FR_I ... max_FR_CA1_E, fout]
    # ----------------------------------------------------
    if (target is None):
        # Defaults
        mu_FR_E = 5
        mu_FR_I = 50
        max_FR_CA1_E = 10
        fout = 6

        # make the goal vector
        goal = [mu_FR_E, mu_FR_I]*4
        goal.append(max_FR_CA1_E)
        goal.append(fout)
    else:
        goal = target

    # Initialize values vector
    # ------------------------
    vals0 = []
    vals = []

    # Firing rates
    # ------------
    # Calcualte FRs per area
    tv_FR, FR_EC_exc, fs_FR = my_FR(spikes=data["EC"]["E"]["t"], duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    _, FR_EC_inh, _ = my_FR(spikes=data["EC"]["I"]["t"], duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    _, FR_DG_exc, _ = my_FR(spikes=data["DG"]["E"]["t"], duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    _, FR_DG_inh, _ = my_FR(spikes=data["DG"]["I"]["t"], duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    _, FR_CA3_exc, _ = my_FR(spikes=data["CA3"]["E"]["t"], duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    _, FR_CA3_inh, _ = my_FR(spikes=data["CA3"]["I"]["t"], duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    _, FR_CA1_exc, _ = my_FR(spikes=data["CA1"]["E"]["t"], duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    _, FR_CA1_inh, _ = my_FR(spikes=data["CA1"]["I"]["t"], duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    # Normalize w.r.t. area size
    FR_EC_exc /= N_EC_exc
    FR_EC_inh /= N_EC_inh
    FR_DG_exc /= N_DG_exc
    FR_DG_inh /= N_DG_inh
    FR_CA3_exc /= N_CA3_exc
    FR_CA3_inh /= N_CA3_inh
    FR_CA1_exc /= N_CA1_exc
    FR_CA1_inh /= N_CA1_inh

    # Mean FR per area
    FR_EC_exc_mean = (len(data["EC"]["E"]["t"])/duration)/N_EC_exc
    FR_EC_inh_mean = (len(data["EC"]["I"]["t"])/duration)/N_EC_inh
    vals0.append(FR_EC_exc_mean)
    vals.append(FR_EC_exc_mean)
    vals.append(FR_EC_inh_mean)

    FR_DG_exc_mean = (len(data["DG"]["E"]["t"])/duration)/N_DG_exc
    FR_DG_inh_mean = (len(data["DG"]["I"]["t"])/duration)/N_DG_inh
    vals0.append(FR_DG_exc_mean)
    vals.append(FR_DG_exc_mean)
    vals.append(FR_DG_inh_mean)

    FR_CA3_exc_mean = (len(data["CA3"]["E"]["t"])/duration)/N_CA3_exc
    FR_CA3_inh_mean = (len(data["CA3"]["I"]["t"])/duration)/N_CA3_inh
    vals0.append(FR_CA3_exc_mean)
    vals.append(FR_CA3_exc_mean)
    vals.append(FR_CA3_inh_mean)

    FR_CA1_exc_mean = (len(data["CA1"]["E"]["t"])/duration)/N_CA1_exc
    FR_CA1_inh_mean = (len(data["CA1"]["I"]["t"])/duration)/N_CA1_inh
    vals0.append(FR_CA1_exc_mean)
    vals.append(FR_CA1_exc_mean)
    vals.append(FR_CA1_inh_mean)

    # Max FR per area
    FR_EC_exc_max = np.max(FR_EC_exc)
    FR_EC_inh_max = np.max(FR_EC_inh)

    FR_DG_exc_max = np.max(FR_DG_exc)
    FR_DG_inh_max = np.max(FR_DG_inh)

    FR_CA3_exc_max = np.max(FR_CA3_exc)
    FR_CA3_inh_max = np.max(FR_CA3_inh)

    FR_CA1_exc_max = np.max(FR_CA1_exc)
    FR_CA1_inh_max = np.max(FR_CA1_inh)
    # vals.append(FR_CA1_exc_max)
    # vals.append(FR_EC_inh_mean)

    # Output Frequency
    pks, _ = sig.find_peaks(data["rhythm"], distance=int(0.100*fs))
    fval = 1/(max(pks[1:] - pks[0:-1])/fs) if len(pks)>1 else 1/(pks[0]/fs)
    # fval = 6
    vals0.append(fval)
    vals.append(fval)

    # Periodogram (PSD)
    # f, PSD = my_PSD()

    # J = g0*A +g1*B + g2*C + g3*D
    J = dst.euclidean(goal, vals0)

    return J, vals


if __name__ == "__main__":
    """ Exhibition of the functions to be used """
    dx = 0.01
    xmin = 5
    xmax = 15
    xv = np.arange(xmin, xmax, dx)

    L0 = L1 = 1
    k0 = k1 = 5.5
    x0 = 8.5
    x1 = 11.

    # Calculations
    y0 = logistic0(xv, L0, k0, x0)
    y1 = logistic1(xv, L1, k1, x1)

    yt0 = my_tanh(xv, gp=1, gn=1, uz=8)
    yt1 = my_tanh(xv, gp=-1, gn=-1, uz=12)

    # Plot the functions
    fig, axs = plt.subplots(3,1, figsize=(12,12))
    axs[0].plot(xv, y0, c='C0', label='L0')
    axs[0].plot(xv, y1, c='C1', label='L1')
    axs[0].plot(xv, y0*y1, c='C3', ls='--', label='L0*L1')
    axs[0].legend(loc=0)

    axs[1].plot(xv, yt0, c='C0', label='tanh')
    axs[1].plot(xv, yt1, c='C1', label='mirrored')
    axs[1].plot(xv, yt0*yt1, c='C3', ls='--', label='product')
    axs[1].legend(loc=0)

    axs[2].plot(xv, my_gauss(xv, mu=10, sigma=1.5), c='C0', label='X~N(10,1.5)')
    axs[2].legend(loc=0)

    # plt.show()


    # Test case
    # ---------
    fs = 10e3
    winsize_FR = 15/1e3
    overlap_FR = 0.9 # percentage
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)

    tdir = 'results/analysis/optimization_test/good_noise/data'
    fnames = ["EC_pyCAN", "EC_inh", "DG_py", "DG_inh", "CA3_pyCAN", "CA3_inh", "CA1_pyCAN", "CA1_inh"]

    data = {}
    for f in fnames:
        tokens = f.split('_')
        area = tokens[0]
        pop = tokens[1]

        if area not in data:
            data[area] = {}
            data[area]["E"] = {}
            data[area]["I"] = {}

        if tokens[1] == "inh":
            data[area]["I"]["t"] = np.loadtxt(tdir + '/spikes/' + f + '_spikemon_t.txt')/1000
            data[area]["I"]["i"] = np.loadtxt(tdir + '/spikes/' + f + '_spikemon_i.txt')/1000
        else:
            data[area]["E"]["t"] = np.loadtxt(tdir + '/spikes/' + f + '_spikemon_t.txt')/1000
            data[area]["E"]["i"] = np.loadtxt(tdir + '/spikes/' + f + '_spikemon_i.txt')/1000

    # Output rhythm
    data["rhythm"] = np.loadtxt(tdir + '/' + 'order_param_mon_rhythm.txt')
    duration = len(data["rhythm"])/fs

    # Run the cost function
    params_FR = {"winsize":winsize_FR, "overlap":overlap_FR}
    J, vec = cost_func(data, None, duration, fs, params_FR=params_FR)

    print("Euclidean distance:", J)
    print(vec)

    # Write to a CSV file
    csv_header = ['config', 'J', 'vector']
    csv_data = ['test0.cfg', J] + vec

    with open('optimization_test.csv', 'a', encoding='UTF8', newline='') as fout:
        writer = csv.writer(fout)

        # write the header
        writer.writerow(csv_header)

        # write the data
        writer.writerow(csv_data)

    exit(0)
