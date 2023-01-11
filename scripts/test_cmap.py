import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(5,5))

# h/v-lines
ax.axhline(y=0, xmin=-1, xmax=1., color='black', linewidth=2)
ax.axvline(x=0, ymin=-1, ymax=1, color='black', linewidth=2)

# Add texts
ax.text(s='Phase Reset', x=0., y=1.25, fontsize=11, transform=ax.transData, va='center', ha='center', clip_on=False)
ax.text(s='Enabled', x=-0.5, y=1.1, fontsize=9, transform=ax.transData, va='center', ha='center', clip_on=False)
ax.text(s='Disabled', x=0.5, y=1.1, fontsize=9, transform=ax.transData, va='center', ha='center', clip_on=False)
ax.text(s='Stimulation', x=-1.25, y=0, fontsize=11, rotation=90, transform=ax.transData, va='center', ha='center', clip_on=False)
ax.text(s='Peak', x=-1.1, y=0.5, fontsize=9, rotation=90, transform=ax.transData, va='center', ha='center', clip_on=False)
ax.text(s='Trough', x=-1.1, y=-0.5, fontsize=9, rotation=90, transform=ax.transData, va='center', ha='center', clip_on=False)

# Colormaps
cmap_spectral = mpl.cm.Spectral
norm_spectral = mpl.colors.Normalize(vmin=-10, vmax=15)

# Use imshow to show the matrix
test_img = np.array([[-10, 5], [10, -5]])
img = ax.imshow(test_img, cmap=cmap_spectral, norm=norm_spectral, extent=[-1,1,-1,1])

# Add colorbar
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(img, cax=cax)

# Set x-ylims
ax.set_ylim([-1,1])
ax.set_xlim([-1,1])

# Show the figure
plt.show()
