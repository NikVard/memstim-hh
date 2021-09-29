#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Memory Stimulation and Phase-Amplitude Coupling
# Copyright 2021 Nikolaos Vardlakis & Amelie Aussel
# Released under the BSD 3-clauses license
# -----------------------------------------------------------------------------

import os
import time
import datetime

import tkinter as tk
from tkinter import ttk

# prevent NumPy from multithreading
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

# Tk window appearance
bg_color = 'white'


# create the GUI
root = tk.Tk() # make the root window
root.title("MemStim - Main Window")
root.minsize(800,400)
root.maxsize(1000,600)
root.config(bg=bg_color)

# notebook style
tab_parent = ttk.Notebook(root)

# make the tabs
tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)

# and connect them to the parent tab
tab_parent.add(tab1, text='Tab 1')
tab_parent.add(tab2, text='Tab 2')

ttk.Label(tab1,
          text ="Tab 1: Parameters setup (JSON load)").grid(column = 0,
                               row = 0,
                               padx = 30,
                               pady = 30)
ttk.Label(tab2,
          text ="Tab 2: Simulation setup").grid(column = 0,
                                    row = 0,
                                    padx = 30,
                                    pady = 30)

# make tabs visible
tab_parent.pack(expand=1, fill="both")

root.mainloop()
