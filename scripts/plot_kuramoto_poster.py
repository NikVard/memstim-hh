# Plot Kuramoto oscillators
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(123)

N = 100 # Number of oscillators
rad = 1. # unit circle

# No sync
theta0 = np.random.uniform(0, 2*np.pi, N) # initial phase
omega0 = np.random.uniform(0, 1, N) # color indicates velocity

X0 = rad*np.cos(theta0)
Y0 = rad*np.sin(theta0)
S0 = 300*np.ones((1,N))
V0 = np.ones((1,N))

# Some sync
theta1 = np.random.uniform(np.pi/2+np.pi/8, 9*np.pi/8, N) # initial phase
omega1 = np.random.uniform(0, 1, N) # color indicates velocity

X1 = rad*np.cos(theta1)
Y1 = rad*np.sin(theta1)
S1 = 300*np.ones((1,N))
V1 = np.ones((1,N))


# Plot
fig, ax = plt.subplots(1,2, figsize=(20,10))
circ0 = plt.Circle((0, 0), radius=rad, edgecolor='k', facecolor='None', lw=0.5)
circ1 = plt.Circle((0, 0), radius=rad, edgecolor='k', facecolor='None', lw=0.5)
ax[0].scatter(X0, Y0, S0, omega0, cmap="twilight", edgecolor="white", zorder=10)
ax[0].add_patch(circ0)
ax[0].set_axis_off()
ax[0].text(-1.2, 1.1, 'A.', fontsize=48)
ax[0].set_xlim([-1.2,1.2])
ax[0].set_ylim([-1.2,1.2])


ax[1].scatter(X1, Y1, S1, omega0, cmap="twilight", edgecolor="white", zorder=10)
ax[1].add_patch(circ1)
ax[1].set_axis_off()
ax[1].text(-1.2, 1.1, 'B.', fontsize=48)
ax[1].set_xlim([-1.2,1.2])
ax[1].set_ylim([-1.2,1.2])

# ax[0].set_rticks([0.25, 0.5, 0.75, 1.])  # Less radial ticks
# ax[0].set_rlabel_position(-22.5)  # Move radial labels away from plotted line

# order parameter
re0 = 1/N * np.sum(np.exp(1j*theta0))
re1 = 1/N * np.sum(np.exp(1j*theta1))
ax[0].arrow(0, 0, np.real(re0), np.imag(re0), alpha=0.5, width=0.015, edgecolor='red', facecolor='red', lw=2, zorder=4, length_includes_head=True, head_length=0.06, head_width=0.06, shape='full')
ax[1].arrow(0, 0, np.real(re1), np.imag(re1), alpha=0.5, width=0.015, edgecolor='red', facecolor='red', lw=2, zorder=4, length_includes_head=True, head_length=0.06, head_width=0.06, shape='full')

ax[0].text(0.2, 0.1, 'r={:.2f}'.format(np.abs(re0)), fontsize=36)
ax[1].text(0.2, 0.1, 'r={:.2f}'.format(np.abs(re1)), fontsize=36)


labels = ['$0\degree$', '$45\degree$', r'$90\degree$', '$135\degree$', r'$180\degree$', '$225\degree$', r'$270\degree$', r'$315\degree$', r'$360\degree$']
pos = np.linspace(0, 2*np.pi, 9)

ax[0].text(1.1*np.cos(pos[0]), 1.*np.sin(pos[0]), labels[0], fontsize=26, ha='left') # 0
ax[0].text(1.1*np.cos(pos[1]), 1.1*np.sin(pos[1]), labels[1], fontsize=26, ha='left') # 45
ax[0].text(1.*np.cos(pos[2]), 1.1*np.sin(pos[2]), labels[2], fontsize=26, ha='center') # 90
ax[0].text(1.1*np.cos(pos[3]), 1.1*np.sin(pos[3]), labels[3], fontsize=26, ha='right') # 135
ax[0].text(1.1*np.cos(pos[4]), 1.*np.sin(pos[4]), labels[4], fontsize=26, ha='right') # 180
ax[0].text(1.1*np.cos(pos[5]), 1.1*np.sin(pos[5]), labels[5], fontsize=26, ha='right') # 225
ax[0].text(1.*np.cos(pos[6]), 1.15*np.sin(pos[6]), labels[6], fontsize=26, ha='center') # 270
ax[0].text(1.1*np.cos(pos[7]), 1.1*np.sin(pos[7]), labels[7], fontsize=26, ha='left') # 315

ax[1].text(1.1*np.cos(pos[0]), 1.*np.sin(pos[0]), labels[0], fontsize=26, ha='left') # 0
ax[1].text(1.1*np.cos(pos[1]), 1.1*np.sin(pos[1]), labels[1], fontsize=26, ha='left') # 45
ax[1].text(1.*np.cos(pos[2]), 1.1*np.sin(pos[2]), labels[2], fontsize=26, ha='center') # 90
ax[1].text(1.1*np.cos(pos[3]), 1.1*np.sin(pos[3]), labels[3], fontsize=26, ha='right') # 135
ax[1].text(1.1*np.cos(pos[4]), 1.*np.sin(pos[4]), labels[4], fontsize=26, ha='right') # 180
ax[1].text(1.1*np.cos(pos[5]), 1.1*np.sin(pos[5]), labels[5], fontsize=26, ha='right') # 225
ax[1].text(1.*np.cos(pos[6]), 1.15*np.sin(pos[6]), labels[6], fontsize=26, ha='center') # 270
ax[1].text(1.1*np.cos(pos[7]), 1.1*np.sin(pos[7]), labels[7], fontsize=26, ha='left') # 315

ax[0].vlines(x=0, ymin=-1., ymax=1., ls='-', color='k', lw=0.75, alpha=0.25)
ax[0].hlines(y=0, xmin=-1., xmax=1., ls='-', color='k', lw=0.75, alpha=0.25)
ax[0].plot(np.linspace(np.sin(3*np.pi/4),np.sin(-np.pi/4),10), np.linspace(-0.707,0.707,10), ls='-', color='k', lw=0.75, alpha=0.25)
ax[0].plot(np.linspace(np.sin(5*np.pi/4),np.sin(np.pi/4),10), np.linspace(-0.707,0.707,10), ls='-', color='k', lw=0.75, alpha=0.25)

ax[1].vlines(x=0, ymin=-1., ymax=1., ls='-', color='k', lw=0.75, alpha=0.25)
ax[1].hlines(y=0, xmin=-1., xmax=1., ls='-', color='k', lw=0.75, alpha=0.25)
ax[1].plot(np.linspace(np.sin(3*np.pi/4),np.sin(-np.pi/4),10), np.linspace(-0.707,0.707,10), ls='-', color='k', lw=0.75, alpha=0.25)
ax[1].plot(np.linspace(np.sin(5*np.pi/4),np.sin(np.pi/4),10), np.linspace(-0.707,0.707,10), ls='-', color='k', lw=0.75, alpha=0.25)


plt.tight_layout()
plt.show();
