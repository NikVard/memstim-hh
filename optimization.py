import os
import warnings
import csv

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.spatial.distance as dst

import parameters
from optlib import cost_func

# Main code
if __name__ == "__main__":
    """ Exhibition of the functions to be used """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Parameter optimization based on simulated spike data')
    parser.add_argument('-a', '--area',
                        nargs='?',
                        metavar='-a',
                        type=str,
                        default=None,
                        help='Default area to load the data from')

    args = parser.parse_args()

    if (args.area is not None):
        basedir = os.path.join('results_opt_'+args.area, 'None')
    else:
        basedir= os.path.join('results_opt', 'None')

    print('Base directory "{0}"'.format(basedir))

    # Parameters
    # ----------
    fnames = ["EC_pyCAN", "EC_inh", "DG_py", "DG_inh", "CA3_pyCAN", "CA3_inh", "CA1_pyCAN", "CA1_inh"]
    fs = 10e3
    winsize_FR = 15/1e3
    overlap_FR = 0.9 # percentage
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)

    # target for EC firing rate w/ noise
    # target_vals = [1., 1., 0., 0., 0., 0., 0., 0., 0., 6.]
    target_vals = [int(args.area == "EC")]*2 + [int(args.area == "DG")]*2 + [int(args.area == "CA3")]*2 + [int(args.area == "CA1")]*2 + [int(args.area == "CA1")] + [6.]

    with open('optimization_' + args.area + '.csv', 'w', encoding='UTF8', newline='') as fout:
        writer = csv.writer(fout)

        # Write the header to the CSV file
        csv_header = ['noise E', 'noise I', 'J', 'vector']
        writer.writerow(csv_header)

        for item in os.listdir(basedir):
            if os.path.isdir(os.path.join(basedir, item)):
                currdir = os.path.join(basedir, item)
                print()
                print('Current directory:', currdir)

                datadir = os.path.join(currdir, 'data')
                spikesdir = os.path.join(datadir, 'spikes')
                # print('Data/Spikes directory:', datadir)

                # Load parameters file for later
                params = parameters.load(os.path.join(currdir, 'parameters_bak.json'))

                data = {}
                for f in fnames:
                    tokens = f.split('_')
                    area = tokens[0]
                    pop = tokens[1]

                    if area not in data:
                        data[area] = {}
                        data[area]["E"] = {}
                        data[area]["I"] = {}

                    # Ignore empty txt file warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        if tokens[1] == "inh":
                            data[area]["I"]["t"] = np.loadtxt(spikesdir + '/' + f + '_spikemon_t.txt', ndmin=1)/1000
                            data[area]["I"]["i"] = np.loadtxt(spikesdir + '/' + f + '_spikemon_i.txt', ndmin=1)/1000
                        else:
                            data[area]["E"]["t"] = np.loadtxt(spikesdir + '/' + f + '_spikemon_t.txt', ndmin=1)/1000
                            data[area]["E"]["i"] = np.loadtxt(spikesdir + '/' + f + '_spikemon_i.txt', ndmin=1)/1000

                        # Output rhythm
                        data["rhythm"] = np.loadtxt(datadir + '/' + 'order_param_mon_rhythm.txt')
                        duration = len(data["rhythm"])/fs

                # Run the cost function
                params_FR = {"winsize":winsize_FR, "overlap":overlap_FR}
                J, vec = cost_func(data, target_vals, duration, fs, params_FR=params_FR)

                print("Euclidean distance:", J)
                print(target_vals)
                print(vec)

                # Write to a CSV file
                csv_data = [params['areas'][args.area]["E"]["noise"], params['areas'][args.area]["I"]["noise"], J] + vec

                print(csv_data)

                # write the data
                writer.writerow(csv_data)

                # continue
                # print("Not reached!")

    exit(0)
