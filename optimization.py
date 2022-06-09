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
    parser.add_argument('-d', '--directory',
                        nargs='?',
                        metavar='-d',
                        type=str,
                        default='results_opt',
                        help='Default directory to load the data from')

    parser.add_argument('-t', '--target',
                        nargs='+',
                        metavar='-t',
                        type=float,
                        default=[6,6,6,6,6],
                        help='Target vector for the optimization; mean FR per area, max FR @ CA1, output frequency')

    parser.add_argument('-o', '--output',
                        nargs='?',
                        metavar='-o',
                        type=str,
                        default='optimization_test.csv',
                        help='Output file name')
    args = parser.parse_args()

    if len(args.target) != 5:
        print("Wrong target length. Try again.")
        exit(-1)

    # Base directory
    basedir = os.path.join(args.directory, 'None')
    print('Base directory "{0}"'.format(basedir))

    # Parameters
    # ----------
    fnames = ["EC_pyCAN", "EC_inh", "DG_py", "DG_inh", "CA3_pyCAN", "CA3_inh", "CA1_pyCAN", "CA1_inh"]
    fs = 10e3
    winsize_FR = 15/1e3
    overlap_FR = 0.9 # percentage
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)

    settling_time = 2 # s
    ending_time = 3 # s

    # target for EC firing rate w/ noise
    # target_vals = [6., 60., 6., 60., 6., 60., 6., 60., 10., 6.]
    # target_vals = [int(args.area == "EC")]*2 + [int(args.area == "DG")]*2 + [int(args.area == "CA3")]*2 + [int(args.area == "CA1")]*2 + [int(args.area == "CA1")] + [6.]
    target_vals = args.target

    with open(args.output, 'w', encoding='UTF8', newline='') as fout:
        writer = csv.writer(fout)

        # Write the header to the CSV file
        csv_header = ['fname', 'J', 'input' ,'a', 'b', 'c', 'd', 'vector']
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
                            t = np.loadtxt(spikesdir + '/' + f + '_spikemon_t.txt', ndmin=1)/1000
                            i = np.loadtxt(spikesdir + '/' + f + '_spikemon_i.txt', ndmin=1)

                            idx_crop = np.where(t <= settling_time)
                            t_tmp = np.delete(t, idx_crop)
                            i_tmp = np.delete(i, idx_crop)
                            data[area]["I"]["t"] = t_tmp
                            data[area]["I"]["i"] = i_tmp

                            idx_crop = np.where(t_tmp >= ending_time)
                            t_tmp = np.delete(t_tmp, idx_crop)
                            i_tmp = np.delete(i_tmp, idx_crop)
                            data[area]["I"]["t"] = t_tmp
                            data[area]["I"]["i"] = i_tmp

                        else:
                            t = np.loadtxt(spikesdir + '/' + f + '_spikemon_t.txt', ndmin=1)/1000
                            i = np.loadtxt(spikesdir + '/' + f + '_spikemon_i.txt', ndmin=1)

                            idx_crop = np.where(t <= settling_time)
                            t_tmp = np.delete(t, idx_crop)
                            i_tmp = np.delete(i, idx_crop)
                            data[area]["E"]["t"] = t_tmp
                            data[area]["E"]["i"] = i_tmp

                            idx_crop = np.where(t_tmp >= ending_time)
                            t_tmp = np.delete(t_tmp, idx_crop)
                            i_tmp = np.delete(i_tmp, idx_crop)
                            data[area]["E"]["t"] = t_tmp
                            data[area]["E"]["i"] = i_tmp

                # Output rhythm
                r = np.loadtxt(datadir + '/' + 'order_param_mon_rhythm.txt')
                data["rhythm"] = r[int(settling_time*fs):int(ending_time*fs)]
                duration = len(data["rhythm"])/fs
                duration0 = (ending_time-settling_time)

                # Run the cost function
                params_FR = {"winsize":winsize_FR, "overlap":overlap_FR}
                J, vec = cost_func(data, target_vals, duration, fs, params_FR=params_FR)

                print("Euclidean distance:", J)
                print(target_vals)
                print(vec)

                # Write to a CSV file
                # csv_data = [os.path.join(currdir, 'parameters_bak.json'), params['areas'][args.area]["E"]["noise"], params['areas'][args.area]["I"]["noise"], J] + vec
                inp_val =  params["Kuramoto"]["gain_rhythm"]
                a = params["connectivity"]["inter_custom"]["EC"]["E"][1][0]
                b = params["connectivity"]["inter_custom"]["EC"]["E"][2][0]
                c = params["connectivity"]["inter_custom"]["EC"]["E"][3][0]
                d = params["connectivity"]["inter_custom"]["CA1"]["E"][0][0]
                noise_EC_exc = params["areas"]["EC"]["E"]["noise"]
                noise_EC_inh = params["areas"]["EC"]["I"]["noise"]
                noise_DG_exc = params["areas"]["DG"]["E"]["noise"]
                noise_DG_inh = params["areas"]["DG"]["I"]["noise"]
                noise_CA3_exc = params["areas"]["CA3"]["E"]["noise"]
                noise_CA3_inh = params["areas"]["CA3"]["I"]["noise"]
                noise_CA1_exc = params["areas"]["CA1"]["E"]["noise"]
                noise_CA1_inh = params["areas"]["CA1"]["I"]["noise"]
                csv_data = [os.path.join(currdir, 'parameters_bak.json'), J, inp_val, a, b, c, d, noise_EC_exc, noise_EC_inh, noise_DG_exc, noise_DG_inh, noise_CA3_exc, noise_CA3_inh, noise_CA1_exc, noise_CA1_inh] + vec

                print(csv_data)

                # write the data
                writer.writerow(csv_data)

                # continue
                # print("Not reached!")

    exit(0)
