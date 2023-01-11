from brian2 import *

# Parameters
areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]
duration = 2*second
dt = 0.1*ms
ts = int(duration/dt)
interp = 'nearest'

platesize = 100
winsize = 10*ms

# Plot in a grid
# fig = plt.figure(figsize=(12,6));
# G = GridSpec(4, 3, width_ratios=(49,49,2))
# ax = np.empty((4,2), dtype=object)


fig, ax = plt.subplots(nrows=4, ncols=2)
fig.set_figheight(12)
fig.set_figwidth(10)


# Loop over files
for ii in range(len(areas)):
    for jj in range(len(areas[ii])):
        t = np.loadtxt('/home/nikos/Documents/projects/Python/memstim-hh/results/analysis/data/spikes/{0}_spikemon_t.txt'.format(areas[ii][jj]))
        i = np.loadtxt('/home/nikos/Documents/projects/Python/memstim-hh/results/analysis/data/spikes/{0}_spikemon_i.txt'.format(areas[ii][jj]))

        N = N_tot[ii][jj] # Neurons

        xedges = np.arange(0,duration/dt,100)
        yedges = np.arange(0,N,50)
        H, xedges, yedges = np.histogram2d(t*ms/dt, i, bins=(xedges,yedges))
        H = H.T
        I = ax[ii,jj].imshow(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis', interpolation=interp, vmin=0, vmax=200, origin='lower', aspect='auto')

        # Set x-y limits - Not working properly, some figures have wrong y-spans (negative values)
        # ax[ii,jj].set_xlim(0, winnum)
        # ax[ii,jj].set_ylim(0, platenum)

        # Set titles and ax labels
        # ax[ii,jj].set_xlabel('Time [s]')
        # ax[ii,jj].set_ylabel('Neurons')

        # We need to draw the canvas, otherwise the labels won't be positioned and
        # won't have values yet.
        fig.canvas.draw()

        # Fix labels to appear in ms
        # xlabels = [item.get_text() for item in ax[ii,jj].get_xticklabels()]
        # ylabels = [item.get_text() for item in ax[ii,jj].get_yticklabels()]
        # xlabels = [int(item)*winsize/second for item in xlabels]
        # ylabels = [int(item)*platesize for item in ylabels]
        ax[ii,jj].set_xticklabels([])
        ax[ii,jj].set_yticklabels([])

        # majors = np.linspace(0, 100, 6)
        # minors = np.linspace(0, duration, 2001)
        # ax.xaxis.set_minor_locator(FixedLocator(minors))


# Colorbar
cbar_ax = fig.add_axes([0.09, 0.05, 0.84, 0.01])
fig.colorbar(I, cax=cbar_ax, orientation='horizontal')#, aspect=1) #shrink=1.0)

# tight_layout()

show()
