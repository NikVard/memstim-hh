import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
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

    plt.show()

    exit(0)
