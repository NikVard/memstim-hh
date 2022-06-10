#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# Parameters
# -----------------------------------------------------------------------------
dt = 0.1e-3
T = 3. # duration, seconds
tv = np.arange(0, T, dt)

t_win = 0.5 # seconds
t_win_s = int(t_win/dt) # samples

t_start = 1.5 # sec
t_start_s = int(t_start/dt)

t_end = t_start + t_win
t_end_s = int(t_end/dt)






# Directories
# -----------------------------------------------------------------------------
dirs = {}

# Results path
dirs['results'] = 'results/movie/'
dirs['data'] = dirs['results'] + 'data/'

# Neuron positions path
dirs['positions'] = dirs['data'] + 'positions/'

# Spikes path
dirs['spikes'] = dirs['data'] + 'spikes/'


# Load the neuron positions
# -----------------------------------------------------------------------------
print(">> Loading positions...")
EC_exc_pos = np.load(dirs['positions'] + 'EC_pyCAN.npy')
EC_inh_pos = np.load(dirs['positions'] + 'EC_inh.npy')
DG_exc_pos = np.load(dirs['positions'] + 'DG_py.npy')
DG_inh_pos = np.load(dirs['positions'] + 'DG_inh.npy')
CA3_exc_pos = np.load(dirs['positions'] + 'CA3_pyCAN.npy')
CA3_inh_pos = np.load(dirs['positions'] + 'CA3_inh.npy')
CA1_exc_pos = np.load(dirs['positions'] + 'CA1_pyCAN.npy')
CA1_inh_pos = np.load(dirs['positions'] + 'CA1_inh.npy')

print(">> Meters to millimeters")
EC_exc_pos *= 1e3
EC_inh_pos *= 1e3
DG_exc_pos *= 1e3
DG_inh_pos *= 1e3
CA3_exc_pos *= 1e3
CA3_inh_pos *= 1e3
CA1_exc_pos *= 1e3
CA1_inh_pos *= 1e3

# Number of E/I neurons per area
EC_exc_N = len(EC_exc_pos)
EC_inh_N = len(EC_inh_pos)
DG_exc_N = len(DG_exc_pos)
DG_inh_N = len(DG_inh_pos)
CA3_exc_N = len(CA3_exc_pos)
CA3_inh_N = len(CA3_inh_pos)
CA1_exc_N = len(CA1_exc_pos)
CA1_inh_N = len(CA1_inh_pos)
N_tot = EC_exc_N + EC_inh_N + DG_exc_N + DG_inh_N + CA3_exc_N + CA3_inh_N + CA1_exc_N + CA1_inh_N

# Load the rasters
print(">> Loading rasters...")
EC_exc_t = np.loadtxt(dirs['spikes'] + 'EC_pyCAN_spikemon_t.txt', dtype=np.float32)
EC_exc_i = np.loadtxt(dirs['spikes'] + 'EC_pyCAN_spikemon_i.txt', dtype=int)
EC_inh_t = np.loadtxt(dirs['spikes'] + 'EC_inh_spikemon_t.txt', dtype=np.float32)
EC_inh_i = np.loadtxt(dirs['spikes'] + 'EC_inh_spikemon_i.txt', dtype=int)

DG_exc_t = np.loadtxt(dirs['spikes'] + 'DG_py_spikemon_t.txt', dtype=np.float32)
DG_exc_i = np.loadtxt(dirs['spikes'] + 'DG_py_spikemon_i.txt', dtype=int)
DG_inh_t = np.loadtxt(dirs['spikes'] + 'DG_inh_spikemon_t.txt', dtype=np.float32)
DG_inh_i = np.loadtxt(dirs['spikes'] + 'DG_inh_spikemon_i.txt', dtype=int)

CA3_exc_t = np.loadtxt(dirs['spikes'] + 'CA3_pyCAN_spikemon_t.txt', dtype=np.float32)
CA3_exc_i = np.loadtxt(dirs['spikes'] + 'CA3_pyCAN_spikemon_i.txt', dtype=int)
CA3_inh_t = np.loadtxt(dirs['spikes'] + 'CA3_inh_spikemon_t.txt', dtype=np.float32)
CA3_inh_i = np.loadtxt(dirs['spikes'] + 'CA3_inh_spikemon_i.txt', dtype=int)

CA1_exc_t = np.loadtxt(dirs['spikes'] + 'CA1_pyCAN_spikemon_t.txt', dtype=np.float32)
CA1_exc_i = np.loadtxt(dirs['spikes'] + 'CA1_pyCAN_spikemon_i.txt', dtype=int)
CA1_inh_t = np.loadtxt(dirs['spikes'] + 'CA1_inh_spikemon_t.txt', dtype=np.float32)
CA1_inh_i = np.loadtxt(dirs['spikes'] + 'CA1_inh_spikemon_i.txt', dtype=int)

# Create the rasters
print(">> Creating raster matrices...")
EC_exc_raster = np.zeros([len(tv), EC_exc_N], dtype=bool)
EC_inh_raster = np.zeros([len(tv), EC_inh_N], dtype=bool)
DG_exc_raster = np.zeros([len(tv), DG_exc_N], dtype=bool)
DG_inh_raster = np.zeros([len(tv), DG_inh_N], dtype=bool)
CA3_exc_raster = np.zeros([len(tv), CA3_exc_N], dtype=bool)
CA3_inh_raster = np.zeros([len(tv), CA3_inh_N], dtype=bool)
CA1_exc_raster = np.zeros([len(tv), CA1_exc_N], dtype=bool)
CA1_inh_raster = np.zeros([len(tv), CA1_inh_N], dtype=bool)

for idx in range(len(EC_exc_t)):
    EC_exc_raster[int(EC_exc_t[idx]*10), EC_exc_i[idx]] = True

for idx in range(len(EC_inh_t)):
    EC_inh_raster[int(EC_inh_t[idx]*10), EC_inh_i[idx]] = True

for idx in range(len(DG_exc_t)):
    DG_exc_raster[int(DG_exc_t[idx]*10), DG_exc_i[idx]] = True

for idx in range(len(DG_inh_t)):
    DG_inh_raster[int(DG_inh_t[idx]*10), DG_inh_i[idx]] = True

for idx in range(len(CA3_exc_t)):
    CA3_exc_raster[int(CA3_exc_t[idx]*10), CA3_exc_i[idx]] = True

for idx in range(len(CA3_inh_t)):
    CA3_inh_raster[int(CA3_exc_t[idx]*10), CA3_inh_i[idx]] = True

for idx in range(len(CA1_exc_t)):
    CA1_exc_raster[int(CA1_exc_t[idx]*10), CA1_exc_i[idx]] = True

for idx in range(len(CA1_inh_t)):
    CA1_inh_raster[int(CA1_inh_t[idx]*10), CA1_inh_i[idx]] = True

# Trim the rasters within the window limits
print(">> Trimming around stimulation window...")
EC_exc_raster_trim = EC_exc_raster[t_idx[0]:t_idx[1]]
EC_inh_raster_trim = EC_inh_raster[t_idx[0]:t_idx[1]]
DG_exc_raster_trim = DG_exc_raster[t_idx[0]:t_idx[1]]
DG_inh_raster_trim = DG_inh_raster[t_idx[0]:t_idx[1]]
CA3_exc_raster_trim = CA3_exc_raster[t_idx[0]:t_idx[1]]
CA3_inh_raster_trim = CA3_inh_raster[t_idx[0]:t_idx[1]]
CA1_exc_raster_trim = CA1_exc_raster[t_idx[0]:t_idx[1]]
CA1_inh_raster_trim = CA1_inh_raster[t_idx[0]:t_idx[1]]

# Load the input rhythm
rhythm = np.loadtxt(dirs['data'] + 'order_param_mon_rhythm.txt')


# Make the figure and render the positions of the neurons
# -----------------------------------------------------------------------------
print(">> Making figure...")
fig = plt.figure(figsize=(8,12), dpi=200)
ax_3D = fig.add_subplot(211, projection='3d')
ax_rhythm = fig.add_subplot(212)

# ax_3D = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax_3D)

# initialization of scatter plots
print(">> Scatter plot initialization...")
# edges = np.zeros((1,4))
# edges[0][3] = 0.25
# edges = 0.
edges = None

faces_py = np.zeros((1,4))
faces_py[0][:3] = [0.25, 0.65, 1.]
faces_py[0][3] = 1.

faces_pyCAN = np.zeros((1,4))
faces_pyCAN[0][:3] = [0.25, 1., .65]
faces_pyCAN[0][3] = 1.

faces_inh = np.zeros((1,4))
faces_inh[0][:3] = [1., 0., 0.5]
faces_inh[0][3] = 1.

EC_exc_scat = ax_3D.scatter(EC_exc_pos[:,0], EC_exc_pos[:,1], EC_exc_pos[:,2], s=1., edgecolors=edges, facecolors=faces_pyCAN, cmap='hot', vmin=0., vmax=1.)
EC_inh_scat = ax_3D.scatter(EC_inh_pos[:,0], EC_inh_pos[:,1], EC_inh_pos[:,2], s=1., edgecolors=edges, facecolors=faces_inh, cmap='hot', vmin=0., vmax=1.)
DG_exc_scat = ax_3D.scatter(DG_exc_pos[:,0], DG_exc_pos[:,1], EC_exc_pos[:,2], s=1., edgecolors=edges, facecolors=faces_py, cmap='hot', vmin=0., vmax=1.)
DG_inh_scat = ax_3D.scatter(DG_inh_pos[:,0], DG_inh_pos[:,1], DG_inh_pos[:,2], s=1., edgecolors=edges, facecolors=faces_inh, cmap='hot', vmin=0., vmax=1.)
CA3_exc_scat = ax_3D.scatter(CA3_exc_pos[:,0], CA3_exc_pos[:,1], CA3_exc_pos[:,2], s=1., edgecolors=edges, facecolors=faces_pyCAN, cmap='hot', vmin=0., vmax=1.)
CA3_inh_scat = ax_3D.scatter(CA3_inh_pos[:,0], CA3_inh_pos[:,1], CA3_inh_pos[:,2], s=1., edgecolors=edges, facecolors=faces_inh, cmap='hot', vmin=0., vmax=1.)
CA1_exc_scat = ax_3D.scatter(CA1_exc_pos[:,0], CA1_exc_pos[:,1], CA1_exc_pos[:,2], s=1., edgecolors=edges, facecolors=faces_pyCAN, cmap='hot', vmin=0., vmax=1.)
CA1_inh_scat = ax_3D.scatter(CA1_inh_pos[:,0], CA1_inh_pos[:,1], CA1_inh_pos[:,2], s=1., edgecolors=edges, facecolors=faces_inh, cmap='hot', vmin=0., vmax=1.)

# Text label to indicate time
# ttext = ax_3D.text(20,10,120,'time=0',fontsize=10)
ttl = ax_3D.text2D(0.05, 0.95, "", transform=ax_3D.transAxes)

# TODO: set figure title


# set axis labels
ax_3D.set_xlabel('X [mm]')
ax_3D.set_ylabel('Y [mm]')
ax_3D.set_zlabel('Z [mm]')

# set axis limits
ax_3D.set_xlim([-3., 12.])
ax_3D.set_ylim([-12., 3.])
ax_3D.set_zlim([0., 15.])

# Also plot the rhythm
tv_rhythm = [t_idx[0]]
ax_rhythm.plot(tv_rhythm, rhythm[t_idx[0]], ls='-', c='k', linewidth=1., rasterized=True, zorder=1)
redDot, = ax_rhythm.plot([t_idx[0]], [rhythm[t_idx[0]]], 'ro')


# Animation functions
# -----------------------------------------------------------------------------
def init():
    global EC_exc_pos, EC_inh_pos, DG_exc_pos, DG_inh_pos, CA3_exc_pos, CA3_inh_pos, CA1_exc_pos, CA1_inh_pos
    global EC_exc_scat, EC_inh_scat, DG_exc_scat, DG_inh_scat, CA3_exc_scat, CA3_inh_scat, CA1_exc_scat, CA1_inh_scat
    global fig, ax_3D
    global tidx, step

    # # Plot the positions of the neurons using scatter
    # EC_exc_scat = ax_3D.scatter(EC_exc_pos[:,0], EC_exc_pos[:,1], EC_exc_pos[:,2], color=[0.6,0.6,0.6], cmap='hot', vmin=0., vmax=1., alpha=0.25)
    # EC_inh_scat= ax_3D.scatter(EC_inh_pos[:,0], EC_inh_pos[:,1], EC_inh_pos[:,2], color=[0.6, 0.6, 0.6], cmap='hot', vmin=0., vmax=1., alpha=0.25)
    # DG_exc_scat = ax_3D.scatter(DG_exc_pos[:,0], DG_exc_pos[:,1], DG_exc_pos[:,2], color=[0.6, 0.6, 0.6], cmap='hot', vmin=0., vmax=1., alpha=0.25)
    # DG_inh_scat = ax_3D.scatter(DG_inh_pos[:,0], DG_inh_pos[:,1], DG_inh_pos[:,2], color=[0.6, 0.6, 0.6], cmap='hot', vmin=0., vmax=1., alpha=0.25)
    # CA3_exc_scat = ax_3D.scatter(CA3_exc_pos[:,0], CA3_exc_pos[:,1], CA3_exc_pos[:,2], color=[0.6, 0.6, 0.6], cmap='hot', vmin=0., vmax=1., alpha=0.25)
    # CA3_inh_scat = ax_3D.scatter(CA3_inh_pos[:,0], CA3_inh_pos[:,1], CA3_inh_pos[:,2], color=[0.6, 0.6, 0.6], cmap='hot', vmin=0., vmax=1., alpha=0.25)
    # CA1_exc_scat = ax_3D.scatter(CA1_exc_pos[:,0], CA1_exc_pos[:,1], CA1_exc_pos[:,2], color=[0.6, 0.6, 0.6], cmap='hot', vmin=0., vmax=1., alpha=0.25)
    # CA1_inh_scat = ax_3D.scatter(CA1_inh_pos[:,0], CA1_inh_pos[:,1], CA1_inh_pos[:,2], color=[0.6, 0.6, 0.6], cmap='hot', vmin=0., vmax=1., alpha=0.25)


    # set camera view
    # ax_3D.view_init(elev=54., azim=-145)

    return EC_exc_scat, EC_inh_scat, DG_exc_scat, DG_inh_scat, CA3_exc_scat, CA3_inh_scat, CA1_exc_scat, CA1_inh_scat


def animate(i):
    global EC_exc_pos, EC_inh_pos, DG_exc_pos, DG_inh_pos, CA3_exc_pos, CA3_inh_pos, CA1_exc_pos, CA1_inh_pos
    global EC_exc_scat, EC_inh_scat, DG_exc_scat, DG_inh_scat, CA3_exc_scat, CA3_inh_scat, CA1_exc_scat, CA1_inh_scat
    global fig, ax_3D
    global tidx, step
    global cf, pf

    # Find neurons that fire at this time
    EC_exc_active = np.where(EC_exc_raster[tidx])
    EC_inh_active = np.where(EC_inh_raster[tidx])
    DG_exc_active = np.where(DG_exc_raster[tidx])
    DG_inh_active = np.where(DG_inh_raster[tidx])
    CA3_exc_active = np.where(CA3_exc_raster[tidx])
    CA3_inh_active = np.where(CA3_inh_raster[tidx])
    CA1_exc_active = np.where(CA1_exc_raster[tidx])
    CA1_inh_active = np.where(CA1_inh_raster[tidx])

    # Update facecolors
    EC_exc_faces = EC_exc_scat.get_facecolors()
    EC_inh_faces = EC_inh_scat.get_facecolors()
    DG_exc_faces = DG_exc_scat.get_facecolors()
    DG_inh_faces = DG_inh_scat.get_facecolors()
    CA3_exc_faces = CA3_exc_scat.get_facecolors()
    CA3_inh_faces = CA3_inh_scat.get_facecolors()
    CA1_exc_faces = CA1_exc_scat.get_facecolors()
    CA1_inh_faces = CA1_inh_scat.get_facecolors()

    # a. Make all colors more transparent as time progresses.
    EC_exc_faces[:,3] -= .1
    EC_exc_faces[:,3] = np.clip(EC_exc_faces[:,3], 0.1, 1)
    EC_inh_faces[:,3] -= .1
    EC_inh_faces[:,3] = np.clip(EC_inh_faces[:,3], 0.1, 1)
    DG_exc_faces[:,3] -= .1
    DG_exc_faces[:,3] = np.clip(DG_exc_faces[:,3], 0.1, 1)
    DG_inh_faces[:,3] -= .1
    DG_inh_faces[:,3] = np.clip(DG_inh_faces[:,3], 0.1, 1)
    CA3_exc_faces[:,3] -= .1
    CA3_exc_faces[:,3] = np.clip(CA3_exc_faces[:,3], 0.1, 1)
    CA3_inh_faces[:,3] -= .1
    CA3_inh_faces[:,3] = np.clip(CA3_inh_faces[:,3], 0.1, 1)
    CA1_exc_faces[:,3] -= .1
    CA1_exc_faces[:,3] = np.clip(CA1_exc_faces[:,3], 0.1, 1)
    CA1_inh_faces[:,3] -= .1
    CA1_inh_faces[:,3] = np.clip(CA1_inh_faces[:,3], 0.1, 1)

    # b. Set the alpha values of the firing neurons to 1.
    EC_exc_faces[EC_exc_active,3] = 1.
    EC_inh_faces[EC_inh_active,3] = 1.
    DG_exc_faces[DG_exc_active,3] = 1.
    DG_inh_faces[DG_inh_active,3] = 1.
    CA3_exc_faces[CA3_exc_active,3] = 1.
    CA3_inh_faces[CA3_inh_active,3] = 1.
    CA1_exc_faces[CA1_exc_active,3] = 1.
    CA1_inh_faces[CA1_inh_active,3] = 1.

    # c. Set the values
    EC_exc_scat.set_facecolors(EC_exc_faces)
    EC_inh_scat.set_facecolors(EC_inh_faces)
    DG_exc_scat.set_facecolors(DG_exc_faces)
    DG_inh_scat.set_facecolors(DG_inh_faces)
    CA3_exc_scat.set_facecolors(CA3_exc_faces)
    CA3_inh_scat.set_facecolors(CA3_inh_faces)
    CA1_exc_scat.set_facecolors(CA1_exc_faces)
    CA1_inh_scat.set_facecolors(CA1_inh_faces)

    # Update marker sizes
    EC_exc_sizes = EC_exc_scat.get_sizes()
    EC_inh_sizes = EC_inh_scat.get_sizes()
    DG_exc_sizes = DG_exc_scat.get_sizes()
    DG_inh_sizes = DG_inh_scat.get_sizes()
    CA3_exc_sizes = CA3_exc_scat.get_sizes()
    CA3_inh_sizes = CA3_inh_scat.get_sizes()
    CA1_exc_sizes = CA1_exc_scat.get_sizes()
    CA1_inh_sizes = CA1_inh_scat.get_sizes()

    # a. Correlate alpha value with size | 0.1 alpha is markersize 1; 1 alpha is markersize 5
    # linear equation f(x)=y -> f(alpha) = size -> f(alpha) = 4.5*alpha + 0.5
    EC_exc_sizes = 4.5 * EC_exc_faces[:,3] + 0.5
    EC_inh_sizes = 4.5 * EC_inh_faces[:,3] + 0.5
    DG_exc_sizes = 4.5 * DG_exc_faces[:,3] + 0.5
    DG_inh_sizes = 4.5 * DG_inh_faces[:,3] + 0.5
    CA3_exc_sizes = 4.5 * CA3_exc_faces[:,3] + 0.5
    CA3_inh_sizes = 4.5 * CA3_inh_faces[:,3] + 0.5
    CA1_exc_sizes = 4.5 * CA1_exc_faces[:,3] + 0.5
    CA1_inh_sizes = 4.5 * CA1_inh_faces[:,3] + 0.5

    # b. Set the marker values
    EC_exc_scat.set_sizes(EC_exc_sizes)
    EC_inh_scat.set_sizes(EC_inh_sizes)
    DG_exc_scat.set_sizes(DG_exc_sizes)
    DG_inh_scat.set_sizes(DG_inh_sizes)
    CA3_exc_scat.set_sizes(CA3_exc_sizes)
    CA3_inh_scat.set_sizes(CA3_inh_sizes)
    CA1_exc_scat.set_sizes(CA1_exc_sizes)
    CA1_inh_scat.set_sizes(CA1_inh_sizes)

    # Update text every 30 frames
    if (i % 30 == 0):
        ctime = int(tidx*dt*1000) # msec
        ttl.set_text('Time = {}ms'.format(ctime))

    # update camera view
    ax_3D.view_init(elev=54., azim=-145+0.05*i)

    # tidx update w/ step
    tidx += step

    return EC_exc_scat, EC_inh_scat, DG_exc_scat, DG_inh_scat, CA3_exc_scat, CA3_inh_scat, CA1_exc_scat, CA1_inh_scat

# Draw once
fig.canvas.draw()

print(">> Starting animation...")
movie = animation.FuncAnimation(fig, animate, frames=F, interval=20, blit=True, init_func=init)
print(">> Movie finished.")

# Save the animation as an mp4 file. Requires ffmpeg encoder.
# The extra arguments given ensure the x264 coden is used (html5 compatibility)
print(">> Exporting as mp4 video file...")
# movie.save(dirs['results'] + 'activation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# print(">> Exporting as gif...")
# movie.save('activation.gif', fps=30)
# print(">> File saved.")

# draw the plot
plt.show()
