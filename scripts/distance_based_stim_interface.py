#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from scipy.spatial import distance as dst
from scipy.spatial.transform import Rotation as R

import tkinter as tk
import tkinter.font as tkFont
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

# scale = round(1000/818,6)*1e-6

class GUI(tk.Tk):
    labels = {}
    labels['areas'] = ['EC', 'DG', 'CA1', 'CA3']
    labels['types'] = ['exc', 'inh']
    labels['coords'] = ['Electrode coordinates', 'x', 'y', 'z']
    labels['slider'] = {'label':'Stimulation Current', 'limits':['-10nA', '10nA']}
    labels['buttons'] = ['Draw']

    # Handle "Enter" events
    def func(self, event):
        print("You hit return.")
        self.plot_structure()

    def plot_structure(self):
        self.ax.cla()

        # reset min/max distance values
        min_, max_ = 0, 0
        sp = []

        # Electrode position
        elec_pos = []
        try:
            # make floats
            x_val = float(self.ex.get())
            y_val = float(self.ey.get())
            z_val = float(self.ez.get())

            # append to list
            elec_pos.append(x_val)
            elec_pos.append(y_val)
            elec_pos.append(z_val)

            # make ndarray
            elec_pos = np.array(elec_pos)[np.newaxis, ...]
        except ValueError:
            traceback.print_exc()
            return

        # Resistivity
        rho = None
        try:
            rho = float(self.rho.get()) # Ohm/mm
        except ValueError:
            traceback.print_exc()
            return

        # Stim amplitude
        I_stim = self.I_stim.get()*1e-9 # nA -> A

        # Add the electrode to the plot
        self.ax.scatter(float(self.ex.get()), float(self.ey.get()), float(self.ez.get()), s=48, c='blue', marker='X', label='Point electrode')

        # Colormap selection
        cmap = 'hot'

        # Downsample plotted neurons
        ds = .25

        # Check and plot
        neuron_pos = self.hipp_coords['EC']['exc']
        idx = np.random.rand(len(neuron_pos)) <= ds
        if self.EC_exc.get():
            d0 = dst.cdist(elec_pos, neuron_pos)
            val = rho*I_stim/(4*np.pi*d0[0,idx])

            if min_ > val.min():
                min_ = val.min()
            if max_ < val.max():
                max_ = val.max()

        else:
            val = 'k'
        self.ax.scatter(neuron_pos[idx,0], neuron_pos[idx,1], neuron_pos[idx,2], c=val, cmap=cmap, marker='o', alpha=.5)

        neuron_pos = self.hipp_coords['EC']['inh']
        idx = np.random.rand(len(neuron_pos)) <= ds
        if self.EC_inh.get():
            d0 = dst.cdist(elec_pos, neuron_pos)
            val = rho*I_stim/(4*np.pi*d0[0,idx])

            if min_ > val.min():
                min_ = val.min()
            if max_ < val.max():
                max_ = val.max()

        else:
            val = 'k'
        self.ax.scatter(neuron_pos[idx,0], neuron_pos[idx,1], neuron_pos[idx,2], c=val, cmap=cmap, marker='^', alpha=.5)

        neuron_pos = self.hipp_coords['DG']['exc']
        idx = np.random.rand(len(neuron_pos)) <= ds
        if self.DG_exc.get():
            d0 = dst.cdist(elec_pos, neuron_pos)
            val = rho*I_stim/(4*np.pi*d0[0,idx])

            if min_ > val.min():
                min_ = val.min()
            if max_ < val.max():
                max_ = val.max()

        else:
            val = 'k'
        self.ax.scatter(neuron_pos[idx,0], neuron_pos[idx,1], neuron_pos[idx,2], c=val, cmap=cmap, marker='s', alpha=.5)

        neuron_pos = self.hipp_coords['DG']['inh']
        idx = np.random.rand(len(neuron_pos)) <= ds
        if self.DG_inh.get():
            d0 = dst.cdist(elec_pos, neuron_pos)
            val = rho*I_stim/(4*np.pi*d0[0,idx])

            if min_ > val.min():
                min_ = val.min()
            if max_ < val.max():
                max_ = val.max()

        else:
            val = 'k'
        self.ax.scatter(neuron_pos[idx,0], neuron_pos[idx,1], neuron_pos[idx,2], c=val, cmap=cmap, marker='^', alpha=.5)

        neuron_pos = self.hipp_coords['CA3']['exc']
        idx = np.random.rand(len(neuron_pos)) <= ds
        if self.CA3_exc.get():
            d0 = dst.cdist(elec_pos, neuron_pos)
            val = rho*I_stim/(4*np.pi*d0[0,idx])

            if min_ > val.min():
                min_ = val.min()
            if max_ < val.max():
                max_ = val.max()

        else:
            val = 'k'
        self.ax.scatter(neuron_pos[idx,0], neuron_pos[idx,1], neuron_pos[idx,2], c=val, cmap=cmap, marker='o', alpha=.5)

        neuron_pos = self.hipp_coords['CA3']['inh']
        idx = np.random.rand(len(neuron_pos)) <= ds
        if self.CA3_inh.get():
            d0 = dst.cdist(elec_pos, neuron_pos)
            val = rho*I_stim/(4*np.pi*d0[0,idx])

            if min_ > val.min():
                min_ = val.min()
            if max_ < val.max():
                max_ = val.max()

        else:
            val = 'k'
        self.ax.scatter(neuron_pos[idx,0], neuron_pos[idx,1], neuron_pos[idx,2], c=val, cmap=cmap, marker='^', alpha=.5)

        neuron_pos = self.hipp_coords['CA1']['exc']
        idx = np.random.rand(len(neuron_pos)) <= ds
        if self.CA1_exc.get():
            d0 = dst.cdist(elec_pos, neuron_pos)
            val = rho*I_stim/(4*np.pi*d0[0,idx])

            if min_ > val.min():
                min_ = val.min()
            if max_ < val.max():
                max_ = val.max()

        else:
            val = 'k'
        self.ax.scatter(neuron_pos[idx,0], neuron_pos[idx,1], neuron_pos[idx,2], c=val, cmap=cmap, marker='o', alpha=.5)

        neuron_pos = self.hipp_coords['CA1']['inh']
        idx = np.random.rand(len(neuron_pos)) <= ds
        if self.CA1_inh.get():
            d0 = dst.cdist(elec_pos, neuron_pos)
            val = rho*I_stim/(4*np.pi*d0[0,idx])

            if min_ > val.min():
                min_ = val.min()
            if max_ < val.max():
                max_ = val.max()

        else:
            val = 'k'
        s0 = self.ax.scatter(neuron_pos[idx,0], neuron_pos[idx,1], neuron_pos[idx,2], c=val, cmap=cmap, marker='^', alpha=.5)


        self.ax.set_title("Anatomical Schematic")
        self.ax.set_xlabel("X [mm]")
        self.ax.set_ylabel("Y [mm]")
        self.ax.set_zlabel("Z [mm]")
        # self.fig.colorbar(s0, ax=self.ax).set_label('Distance')

        # Custom legend entries
        custom_lines = [Line2D([0], [0], ls='', marker='o', color='k', markersize=10, label='Py-CAN'),
                        Line2D([0], [0], ls='', marker='s', color='k', markersize=10, label='Py-non-CAN'),
                        Line2D([0], [0], ls='', marker='^', color='k', markersize=10, label='Inh. interneurons')]

        handles, labels = self.ax.get_legend_handles_labels()
        handles.extend(custom_lines)
        self.ax.legend(handles=handles)

        # TODO: add colorbar w/ limits


        self.canvas.draw()


    def __init__(self, struct_coords):
        super().__init__()
        self.title('3D Hippocampal Formation')
        # self.geometry('1000x800')
        self.minsize(800, 800)
        self.resizable(False, False)

        # layout all of main containers, weight applied on the figure
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)


        # Instance variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # electrode coordinates <float>
        self.xc = tk.DoubleVar()
        self.yc = tk.DoubleVar()
        self.zc = tk.DoubleVar()

        # stim. amplitude <float>
        self.I_stim = tk.DoubleVar()

        # rho <float>
        self.rho = tk.DoubleVar()

        # stim on/off <bool>
        self.EC_exc = tk.BooleanVar()
        self.EC_inh = tk.BooleanVar()
        self.DG_exc = tk.BooleanVar()
        self.DG_inh = tk.BooleanVar()
        self.CA3_exc = tk.BooleanVar()
        self.CA3_inh = tk.BooleanVar()
        self.CA1_exc = tk.BooleanVar()
        self.CA1_inh = tk.BooleanVar()

        # Initialization of variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.hipp_coords = struct_coords
        self.xc.set(0.)
        self.yc.set(0.)
        self.zc.set(0.)

        self.I_stim.set(0.)

        self.rho.set(0.)

        self.EC_exc.set(True)
        self.EC_inh.set(True)
        self.DG_exc.set(False)
        self.DG_inh.set(False)
        self.CA3_exc.set(False)
        self.CA3_inh.set(False)
        self.CA1_exc.set(False)
        self.CA1_inh.set(False)


        # Frames
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        __plotFrame = tk.Frame(self, width=700, height=500, background='blue')
        __plotFrame.grid(row=0, column=0, padx=5, pady=2, sticky=tk.NW+tk.NE+tk.S)

        __controlsFrame1 = tk.Frame(self, width=700, height=300, background='red')
        __controlsFrame1.grid(row=1, column=0, padx=5, pady=2, sticky=tk.SW+tk.SE)

        __controlsFrame2 = tk.Frame(self, width=300, height=800, background='green')
        __controlsFrame2.grid(row=0, column=1, rowspan=2, padx=5, pady=2, sticky=tk.NE+tk.SE)


        # Plot figure
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # the figure that will contain the plot
        self.fig = Figure(figsize=(6,6), dpi = 128)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # creating the Tkinter canvas containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master = __plotFrame)
        self.canvas.get_tk_widget().grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        self.canvas.draw()

        # __toolbarFrame = tk.Frame(master=__plotFrame)
        # self.toolbar = NavigationToolbar2Tk(self.canvas, __toolbarFrame)
        # self.toolbar.update()
        # self.toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        # Controls #1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # layout the containers, weight applied on the slider
        __controlsFrame1.grid_rowconfigure(1, weight=1)
        __controlsFrame1.grid_columnconfigure(0, weight=1)

        # Electrode stimulation strength
        stimLabel = tk.Label(__controlsFrame1, text='Stimulation current Amplitude [nA]')
        self.stimSlider = tk.Scale(__controlsFrame1, from_=-10., to=10., resolution = 0.5, length=500, tickinterval=.5, orient=tk.HORIZONTAL, variable=self.I_stim)

        stimLabel.grid(row=0, column=0, sticky="", padx=2, pady=5)
        self.stimSlider.grid(row=1, column=0, sticky="", padx=2, pady=5)

        # Sigma [S/cm]
        condLabel = tk.Label(__controlsFrame1, text=r'Medium conductivity $\sigma$ [S/cm]')
        self.condEntry = tk.Entry(__controlsFrame1, width=5, textvariable=self.rho)

        condLabel.grid(row=0, column=1, sticky="", padx=2, pady=5)
        self.condEntry.grid(row=1, column=1, sticky="", padx=2, pady=5)


        # Controls #2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        __controlsFrame2.grid_columnconfigure(0, weight=1)

        # Electrode coordinates
        __coordLabelsFrame = tk.Frame(master=__controlsFrame2, background='orange')
        # __coordLabelsFrame.grid(row=0, column=0, padx=5, pady=2, sticky='nsew')
        __coordLabelsFrame.pack(pady=20)

        labelCoords = tk.Label(__coordLabelsFrame, text='Electrode coordinates [mm]')
        labelX = tk.Label(__coordLabelsFrame, text='x')
        labelY = tk.Label(__coordLabelsFrame, text='y')
        labelZ = tk.Label(__coordLabelsFrame, text='z')
        self.ex = tk.Entry(__coordLabelsFrame, width=5, textvariable=self.xc)
        self.ey = tk.Entry(__coordLabelsFrame, width=5, textvariable=self.yc)
        self.ez = tk.Entry(__coordLabelsFrame, width=5, textvariable=self.zc)

        # Label
        labelCoords.grid(row=0, column=0, columnspan=3, sticky=tk.W + tk.E, pady=2)
        __controlsFrame2.grid_rowconfigure(0, minsize=20)

        # X
        labelX.grid(row=1, column=0, pady=2)
        self.ex.grid(row=2, column=0, padx=2, pady=2)

        # Y
        labelY.grid(row=1, column=1, pady=2)
        self.ey.grid(row=2, column=1, padx=2, pady=2)

        # Z
        labelZ.grid(row=1, column=2, pady=2)
        self.ez.grid(row=2, column=2, padx=2, pady=2)

        # Stim. areas
        __stimAreasFrame = tk.Frame(master=__controlsFrame2, background='yellow')
        # __stimAreasFrame.grid(row=1, column=0, padx=5, pady=10, sticky='nsew')
        __stimAreasFrame.pack(fill=tk.X, expand=False, pady=20)

        labelStimAreas = tk.Label(__stimAreasFrame, text='Stimulation Areas')
        labelEC = tk.Label(__stimAreasFrame, text='EC')
        labelDG = tk.Label(__stimAreasFrame, text='DG')
        labelCA3 = tk.Label(__stimAreasFrame, text='CA3')
        labelCA1 = tk.Label(__stimAreasFrame, text='CA1')

        self.cbECexc = tk.Checkbutton(__stimAreasFrame, text="E", variable=self.EC_exc)
        self.cbECinh = tk.Checkbutton(__stimAreasFrame, text="I", variable=self.EC_inh)
        self.cbDGexc = tk.Checkbutton(__stimAreasFrame, text="E", variable=self.DG_exc)
        self.cbDGinh = tk.Checkbutton(__stimAreasFrame, text="I", variable=self.DG_inh)
        self.cbCA3exc = tk.Checkbutton(__stimAreasFrame, text="E", variable=self.CA3_exc)
        self.cbCA3inh = tk.Checkbutton(__stimAreasFrame, text="I", variable=self.CA3_inh)
        self.cbCA1exc = tk.Checkbutton(__stimAreasFrame, text="E", variable=self.CA1_exc)
        self.cbCA1inh = tk.Checkbutton(__stimAreasFrame, text="I", variable=self.CA1_inh)

        # then place in grid
        __stimAreasFrame.grid_rowconfigure(0, weight=1)
        __stimAreasFrame.grid_columnconfigure(0, weight=1)
        __stimAreasFrame.grid_columnconfigure(1, weight=1)
        __stimAreasFrame.grid_columnconfigure(2, weight=1)
        __stimAreasFrame.grid_columnconfigure(3, weight=1)

        labelStimAreas.grid(row=0, column=0, columnspan=4, sticky=tk.W + tk.E, pady=2)
        labelEC.grid(row=1, column=0, columnspan=1)
        labelDG.grid(row=1, column=1, columnspan=1)
        self.cbECexc.grid(row=2, column=0)
        self.cbECinh.grid(row=3, column=0)
        self.cbDGexc.grid(row=2, column=1)
        self.cbDGinh.grid(row=3, column=1)

        # __controlsFrame2.grid_rowconfigure(5, minsize=25)
        labelCA3.grid(row=1, column=2, columnspan=1)
        labelCA1.grid(row=1, column=3, columnspan=1)
        self.cbCA3exc.grid(row=2, column=2)
        self.cbCA3inh.grid(row=3, column=2)
        self.cbCA1exc.grid(row=2, column=3)
        self.cbCA1inh.grid(row=3, column=3)


        # Update draw button
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        __updateBtnFrame = tk.Frame(master=__controlsFrame2, background='cyan')
        # __updateBtnFrame.grid(row=2, column=0, padx=5, pady=10)
        __updateBtnFrame.pack(pady=20)
        self.btnUpdate = tk.Button(__updateBtnFrame, text='Update', command=self.plot_structure)
        self.btnUpdate.grid(row=0, column=0)


        self.bind('<Return>', self.func)


# Main script
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__== '__main__':

    # import scipy rotations
    from scipy.spatial.transform import Rotation as R

    scale = round(1000/818,6)*1e-3

    # Make a function to read the data from the numpy files
    def import_data(fname):
        try:
            pos = np.load(fname)
        except Exception as e:
            print(e)
            print('Exception caught, returning...')
            return None

        # rotation matrix for fixing rotated y-positions from stippling program
        r = R.from_euler('x', 180, degrees=True)

        # read and transform the coordinates
        pos = np.hstack((pos, np.zeros((len(pos), 1))))
        pos = r.apply(pos)
        pos *= scale
        pos[:,2] += 15*np.random.rand(len(pos))
        idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
        pos = pos[idx]

        return pos

    # EC
    EC_E = import_data('./neuron_positions/full/EC_E-stipple-10000.npy')
    EC_I = import_data('./neuron_positions/full/EC_I-stipple-1000.npy')
    if (EC_E is None) or (EC_I is None):
        print("Error: Some coordinates (EC) are not loaded. Exiting...")
        exit(-1)

    # DG
    DG_E = import_data('./neuron_positions/full/DG_E-stipple-10000.npy')
    DG_I = import_data('./neuron_positions/full/DG_I-stipple-100.npy')
    if (DG_E is None) or (DG_I is None):
        print("Error: Some coordinates (DG) are not loaded. Exiting...")
        exit(-1)

    # CA3
    CA3_E = import_data('./neuron_positions/full/CA3_E-stipple-1000.npy')
    CA3_I = import_data('./neuron_positions/full/CA3_I-stipple-100.npy')
    if (CA3_E is None) or (CA3_I is None):
        print("Error: Some coordinates (CA3) are not loaded. Exiting...")
        exit(-1)

    # CA1
    CA1_E = import_data('./neuron_positions/full/CA1_E-stipple-10000.npy')
    CA1_I = import_data('./neuron_positions/full/CA1_I-stipple-1000.npy')
    if (CA1_E is None) or (CA1_I is None):
        print("Error: Some coordinates (CA1) are not loaded. Exiting...")
        exit(-1)

    struct_coords = dict(EC = {"exc" : EC_E, "inh" : EC_I},
                         DG = {"exc" : DG_E, "inh" : DG_I},
                         CA3 = {"exc" : CA3_E, "inh" : CA3_I},
                         CA1 = {"exc" : CA1_E, "inh" : CA1_I})

    gui = GUI(struct_coords)
    gui.mainloop()
