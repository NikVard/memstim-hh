#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Memory Stimulation and Phase-Amplitude Coupling
# Copyright 2021 Nikolaos Vardlakis & Nicolas P. Rougier
# Released under the BSD 3-clauses license
# -----------------------------------------------------------------------------
import os
import json
import time
import subprocess
import numpy as np
from numpy import pi

# Default parameters
noise_EC = noise_DG = noise_CA3 = noise_CA1 = 0.
a = b = c = d = 0. # connections
a = 0.
b = 0.
c = 0.
d = 0.
I_in = 0.0 # input

_data = {
    "seed_val"  : 42,       # Reproducibility

    # areas, tables 3.1-3.3, pages 45-48, Aussel
    "areas": {
        "EC"    : {
            "E" : {
                "N" : int(10e3),
                "type" : "PyCAN",
                "noise" : np.around(noise_EC * 100.e-06, 6)        # Volts
            },
            "I" : {
                "N" : int(1e3),
                "type" : "Inh",
                "noise" : np.around(noise_EC * 10.e-06, 6)
            }
        },
        "DG"    : {
            "E" : {
                "N" : int(10e3),
                "type" : "Py",
                "noise" : np.around(noise_DG * 100.e-06, 6)
            },
            "I" : {
                "N" : int(0.1e3),
                "type" : "Inh",
                "noise" : np.around(noise_DG * 10.e-06, 6)
            }
        },
        "CA3"   : {
            "E" : {
                "N" : int(1e3),
                "type" : "PyCAN",
                "noise" : np.around(noise_CA3 * 100.e-06, 6)
            },
            "I" : {
                "N" : int(0.1e3),
                "type" : "Inh",
                "noise" : np.around(noise_CA3 * 10.e-06, 6)
            }
        },
        "CA1"   : {
            "E" : {
                "N" : int(10e3),
                "type" : "PyCAN",
                "noise" : np.around(noise_CA1 * 100.e-06, 6)
            },
            "I" : {
                "N" : int(1e3),
                "type" : "Inh",
                "noise" : np.around(noise_CA1 * 10.e-06, 6)
            }
        }
    },

    # Kuramoto oscillator parameters
    "Kuramoto" : {
        "N" : 200,
        "f0" : 6.,
        "sigma" : 0.5,  # normal std
        "kN" : 20,
        "gain_reset" : 0,
        "gain_rhythm" : np.around(I_in, 2), # nA
        "offset" : -0*pi/2
    },

    # connectivity parameters
    "connectivity" : {
        "intra" : { # intra-area conn. probabilities per area |
            "EC" : [[0., 0.37], [0.54, 0.]], # [[E-E, E-I], [I-E, I-I]]
            "DG" : [[0., 0.06], [0.14, 0.]],
            "CA3" : [[0.56, 0.75], [0.75, 0.]],
            "CA1" : [[0., 0.28], [0.3, 0.7]]
        },
        "inter" : { # inter-area conn. probabilities
            "p_tri" : 0.45, # tri: [DG->CA3, CA3->CA1, CA1->EC] Aussel, pages 49,59
            "p_mono" : 0.3  # mono: [EC->CA3, EC->CA1]
        },
        "inter_custom" : {
            "EC" : {
                "E" : [[0., 0.], [a, a], [b, b], [c, c]],
                "I" : [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]
            },
            "DG" : {
                "E" : [[0., 0.], [0., 0.], [b, b], [0., 0.]],
                "I" : [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]
            },
            "CA3" : {
                "E" : [[0., 0.], [0., 0.], [0., 0.], [c, c]],
                "I" : [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]
            },
            "CA1" : {
                "E" : [[d, d], [0., 0.], [0., 0.], [0., 0.]],
                "I" : [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]
            }
        }
    },

    # synapses
    "synapses" : {
        "gmax_e" : 600.,    # pSiemens
        "gmax_i" : 60.
    },

    # stimulation parameters
    "stimulation" : {
        "target" : "CA1",                   # target area [EC | DG | CA3 | CA1]
        "coordinates" : (5.0, -8., 7.5),    # point electrode coordinates (x,y,z) [mm]
        "rho" : 1.,                         # resistivity of homogeneous conductive medium [Ω/cm]
        "duration" : 2.,                    # [sec]
        "dt" : .1e-3,                       # [sec]
        "onset" : 0.50,                     # [sec]
        "I" : [0.],                         # stimulation amplitude [nA]
        "pulse_width" : [1.e-3],            # width (in time) of pulse ON phase [sec]
        "stim_freq" : 5,                    # stimulation frequency [Hz]
        "pulse_freq" : 100,                 # pulse frequency, determines ON duration [Hz]
        "nr_of_trains" : 1,                 # number of pulse trains
        "nr_of_pulses" : 1,                 # number of pulses per train
        "ipi" : .1e-3                       # inter-pulse interval [sec]
        },

    # simulation parameters
    "simulation" : {
        "duration"  : 1.0,                  # second
        "dt"        : .1e-3,                # second
        "debugging" : False
    },

    # git stuffsubprocess
    "timestamp"         : None,
    "git_branch"        : None,
    "git_hash"          : None,
    "git_short_hash"    : None
}

def is_git_repo():
    """ Return whether current directory is a git directory """
    if subprocess.call(["git", "branch"],
            stderr=subprocess.STDOUT, stdout=open(os.devnull, 'w')) != 0:
        return False
    return True

def get_git_revision_hash():
    """ Get current git hash """
    if is_git_repo():
        answer = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'])
        return answer.decode("utf8").strip("\n")
    return "None"

def get_git_revision_short_hash():
    """ Get current git short hash """
    if is_git_repo():
        answer = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'])
        return answer.decode("utf8").strip("\n")
    return "None"

def get_git_revision_branch():
    """ Get current git branch """
    if is_git_repo():
        answer = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        return answer.decode("utf8").strip("\n")
    return "None"

def default():
    """ Get default parameters """
    _data["timestamp"] = time.ctime()
    _data["git_branch"] = get_git_revision_branch()
    _data["git_hash"] = get_git_revision_hash()
    _data["git_short_hash"] = get_git_revision_short_hash()
    return _data

def save(filename, data=None):
    """ Save parameters into a json file """
    if data is None:
       data = { name : eval(name) for name in _data.keys()
                if name not in ["timestamp", "git_branch", "git_hash"] }
    data["timestamp"] = time.ctime()
    data["git_branch"] = get_git_revision_branch()
    data["git_hash"] = get_git_revision_hash()
    data["git_short_hash"] = get_git_revision_short_hash()
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=False)

def load(filename):
    """ Load parameters from a json file """
    with open(filename) as infile:
        data = json.load(infile)
    return data

def dump(data):
    if not _data["timestamp"]:
        _data["timestamp"] = time.ctime()
    if not _data["git_branch"]:
        _data["git_branch"] = get_git_revision_branch()
    if not _data["git_hash"]:
        _data["git_hash"] = get_git_revision_hash()
        _data["git_short_hash"] = get_git_revision_short_hash()
    for key, value in data.items():
        print(f"{key:15s} : {value}")

# -----------------------------------------------------------------------------
if __name__  == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate parameters file using JSON format')
    parser.add_argument('-f', '--filename',
                        default='default',
                        type=str, nargs='?',
                        help='Parameters file (json format)')

    parser.add_argument('-o', '--output-directory',
                        nargs='?',
                        type=str,
                        default='configs',
                        help='Output directory')

    args = parser.parse_args()

    # Make directories
    basedir = args.output_directory
    if not os.path.isdir(basedir):
        print('[+] Creating directory', basedir)
        os.makedirs(basedir)

    # dir_EC = os.path.join(basedir, 'opt_noise_EC')
    # dir_DG = os.path.join(basedir, 'opt_noise_DG')
    # dir_CA3 = os.path.join(basedir, 'opt_noise_CA3')
    # dir_CA1 = os.path.join(basedir, 'opt_noise_CA1')

    vmin,vmax = 30,45
    noise_vals = np.arange(vmin, vmax)

    areas = ['EC', 'DG', 'CA3', 'CA1']
    # for area in areas:
    #     currdir = os.path.join(basedir, 'opt_noise_'+area)
    #     if not os.path.isdir(currdir):
    #         print('[+] Creating directory', currdir)
    #         os.makedirs(currdir)
    #
    #     # Fix the parameters
    #     cnt=0
    #     for val in noise_vals:
    #         _data["areas"][area]["E"]["noise"] = np.around(val*100.e-06, 6)
    #         _data["areas"][area]["I"]["noise"] = np.around(val*10.e-06, 6)
    #
    #         # Define the filename
    #         # filename = "./{0}/{1}_{2:02d}.json".format(args.output_directory, args.filename, cnt)
    #         filename = os.path.join(currdir, (args.filename+'{0:02d}.json').format(cnt))
    #         print('Saving file "{0}"'.format(filename))
    #         save(filename, _data)
    #         cnt += 1
    #
    #         _data["areas"][area]["E"]["noise"] = 0.
    #         _data["areas"][area]["I"]["noise"] = 0.

    vmin,vmax = 0.1, 1.3
    vals = np.arange(vmin, vmax, 0.1)
    cnt = 0
    for val in vals:
        _data["Kuramoto"]["gain_rhythm"] = np.around(val, 2)
        # _data["connectivity"]["inter_custom"]["EC"]["E"][1] = [np.around(val, 1)]*2

        # Define the filename
        filename = os.path.join(basedir, (args.filename+'{0:02d}.json').format(cnt))
        print('Saving file "{0}"'.format(filename))
        save(filename, _data)
        cnt += 1


    # _data["connectivity"]["inter_custom"]["EC"]["E"][1] = np.around([a, a], decimals=1).tolist()
    # _data["connectivity"]["inter_custom"]["EC"]["E"][2] = np.around([b, b], decimals=1).tolist()
    # _data["connectivity"]["inter_custom"]["EC"]["E"][3] = np.around([c, c], decimals=1).tolist()
    # _data["connectivity"]["inter_custom"]["DG"]["E"][2] = np.around([b, b], decimals=1).tolist()
    # _data["connectivity"]["inter_custom"]["CA3"]["E"][3] = np.around([c, c], decimals=1).tolist()
    # _data["connectivity"]["inter_custom"]["CA1"]["E"][0] = np.around([d, d], decimals=1).tolist()
