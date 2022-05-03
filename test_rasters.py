from brian2 import *
import os

def parse_file_t(fname):
    """ Parses Amelie's saved rasters, t files """
    # initialize read data structure
    data = []

    with open(fname) as f:
        lines = f.readlines()

        if lines:
            # tokenize, split the data at ","
            tokens = lines[0].split(',')

            # split the string at space " "
            for tok in tokens:
                tmp = tok.split(' ')
                if len(tmp) == 2:
                    val = tmp[0]
                    unit = tmp[1]

                    if unit == 'ms':
                        data.append(float(val)/1000)
                    else:
                        data.append(float(val))
    return array(data)

def parse_file_i(fname):
    """ Parses Amelie's saved rasters, i files """
    # initialize read data structure
    data = []

    with open(fname) as f:
        lines = f.readlines()

        if lines:
            # tokenize, split at ","
            tokens = lines[0].split(',')

            for tok in tokens:
                if tok != '':
                    data.append(int(tok))
    return array(data)


# Rasters directory
# rastdir_A = "/home/nikos/Documents/projects/Python/aussel/healthy_and_epileptic_hippocampus_model/results_2022-05-01 21:37:51.456528/rasters/"
# rastdir_A = "/home/nikos/Documents/projects/Python/aussel/healthy_and_epileptic_hippocampus_model/test_results_2022-05-02 11:51:36.217288/rasters/" # G=1, 6Hz @ EC E/I, EC intra=0, ptri=pmono=0
rastdir_A = "/home/nikos/Documents/projects/Python/aussel/healthy_and_epileptic_hippocampus_model/results_2022-05-03 14:32:20.059510/rasters/"
# rastdir_B = "/home/nikos/Documents/projects/Python/memstim-hh/results_test_DC/None/02-05-2022 12H23M32S/data/spikes/" # G=1, 6Hz @ EC E/I, EC intra=0, ptri=pmono=0
rastdir_B = "/home/nikos/Documents/projects/Python/memstim-hh/results/None/03-05-2022 14H14M53S/data/spikes/"
# rastdir_C = "/home/nikos/Documents/projects/Python/memstim-hh/results_Amelie_eqs/None/28-04-2022 16H49M54S/data/spikes/"
rastdir_C = "/home/nikos/Documents/projects/Python/memstim-hh/results/None/03-05-2022 14H42M39S/data/spikes/"

# Load the rasters
rast_EC_E_t = parse_file_t(os.path.join(rastdir_A, "raster_EC_exc_t.txt"))
rast_EC_E_i = parse_file_i(os.path.join(rastdir_A, "raster_EC_exc_i.txt"))
rast_EC_I_t = parse_file_t(os.path.join(rastdir_A, "raster_EC_inh_t.txt"))
rast_EC_I_i = parse_file_i(os.path.join(rastdir_A, "raster_EC_inh_i.txt"))
rast_DG_E_t = parse_file_t(os.path.join(rastdir_A, "raster_DG_exc_t.txt"))
rast_DG_E_i = parse_file_i(os.path.join(rastdir_A, "raster_DG_exc_i.txt"))
rast_DG_I_t = parse_file_t(os.path.join(rastdir_A, "raster_DG_inh_t.txt"))
rast_DG_I_i = parse_file_i(os.path.join(rastdir_A, "raster_DG_inh_i.txt"))
rast_CA3_E_t = parse_file_t(os.path.join(rastdir_A, "raster_CA3_exc_t.txt"))
rast_CA3_E_i = parse_file_i(os.path.join(rastdir_A, "raster_CA3_exc_i.txt"))
rast_CA3_I_t = parse_file_t(os.path.join(rastdir_A, "raster_CA3_inh_t.txt"))
rast_CA3_I_i = parse_file_i(os.path.join(rastdir_A, "raster_CA3_inh_i.txt"))
rast_CA1_E_t = parse_file_t(os.path.join(rastdir_A, "raster_CA1_exc_t.txt"))
rast_CA1_E_i = parse_file_i(os.path.join(rastdir_A, "raster_CA1_exc_i.txt"))
rast_CA1_I_t = parse_file_t(os.path.join(rastdir_A, "raster_CA1_inh_t.txt"))
rast_CA1_I_i = parse_file_i(os.path.join(rastdir_A, "raster_CA1_inh_i.txt"))

# Area sizes
# N_EC_E = round(max(rast_EC_E_i),-2)
# N_EC_I = round(max(rast_EC_I_i),-2)
# N_DG_E = round(max(rast_DG_E_i),-2)
# N_DG_I = round(max(rast_DG_I_i),-2)
# N_CA3_E = round(max(rast_CA3_E_i),-2)
# N_CA3_I = round(max(rast_CA3_I_i),-2)
# N_CA1_E = round(max(rast_CA1_E_i),-3)
# N_CA1_I = round(max(rast_CA1_I_i),-2)

N_EC_E = 10000
N_EC_I = 1000
N_DG_E = 10000
N_DG_I = 100
N_CA3_E = 1000
N_CA3_I = 100
N_CA1_E = 10000
N_CA1_I = 1000


# print("N_EC: ", N_EC_E, N_EC_I)
# print("N_DG: ", N_DG_E, N_DG_I)
# print("N_CA3: ", N_CA3_E, N_CA3_I)
# print("N_CA1: ", N_CA1_E, N_CA1_I)

# Mean FRs
FR_EC_E_mean = ((len(rast_EC_E_t)/(max(rast_EC_E_t)-min(rast_EC_E_t)))/N_EC_E)
FR_EC_I_mean = ((len(rast_EC_I_t)/(max(rast_EC_I_t)-min(rast_EC_I_t)))/N_EC_I)
FR_DG_E_mean = ((len(rast_DG_E_t)/(max(rast_DG_E_t)-min(rast_DG_E_t)))/N_DG_E)
FR_DG_I_mean = ((len(rast_DG_I_t)/(max(rast_DG_I_t)-min(rast_DG_I_t)))/N_DG_I)
FR_CA3_E_mean = ((len(rast_CA3_E_t)/(max(rast_CA3_E_t)-min(rast_CA3_E_t)))/N_CA3_E)
FR_CA3_I_mean = ((len(rast_CA3_I_t)/(max(rast_CA3_I_t)-min(rast_CA3_I_t)))/N_CA3_I)
FR_CA1_E_mean = ((len(rast_CA1_E_t)/(max(rast_CA3_E_t)-min(rast_CA3_E_t)))/N_CA1_E)
FR_CA1_I_mean = ((len(rast_CA1_I_t)/(max(rast_CA1_I_t)-min(rast_CA1_I_t)))/N_CA1_I)

# Print results
print("Mean FRs - Amelie's code")
print("EC:", "\t", round(FR_EC_E_mean,4), "\t", round(FR_EC_I_mean,4))
# print("EC E, tmin:", min(rast_EC_E_t), "tmax:", max(rast_EC_E_t))
# print("EC I, tmin:", min(rast_EC_I_t), "tmax:", max(rast_EC_I_t))
print("DG:", "\t", round(FR_DG_E_mean,4), "\t", round(FR_DG_I_mean,4))
print("CA3:", "\t", round(FR_CA3_E_mean,4), "\t", round(FR_CA3_I_mean,4))
print("CA1:", "\t", round(FR_CA1_E_mean,4), "\t", round(FR_CA1_I_mean,4))
print()

# My turn
# +=================================+
# Nicolas' positions
# +=================================+
# Load the rasters
rast_EC_E_t = loadtxt(os.path.join(rastdir_B, "EC_pyCAN_spikemon_t.txt"))
rast_EC_E_i = loadtxt(os.path.join(rastdir_B, "EC_pyCAN_spikemon_i.txt"))
rast_EC_I_t = loadtxt(os.path.join(rastdir_B, "EC_inh_spikemon_t.txt"))
rast_EC_I_i = loadtxt(os.path.join(rastdir_B, "EC_inh_spikemon_i.txt"))
rast_DG_E_t = loadtxt(os.path.join(rastdir_B, "DG_py_spikemon_t.txt"))
rast_DG_E_i = loadtxt(os.path.join(rastdir_B, "DG_py_spikemon_i.txt"))
rast_DG_I_t = loadtxt(os.path.join(rastdir_B, "DG_inh_spikemon_t.txt"))
rast_DG_I_i = loadtxt(os.path.join(rastdir_B, "DG_inh_spikemon_i.txt"))
rast_CA3_E_t = loadtxt(os.path.join(rastdir_B, "CA3_pyCAN_spikemon_t.txt"))
rast_CA3_E_i = loadtxt(os.path.join(rastdir_B, "CA3_pyCAN_spikemon_i.txt"))
rast_CA3_I_t = loadtxt(os.path.join(rastdir_B, "CA3_inh_spikemon_t.txt"))
rast_CA3_I_i = loadtxt(os.path.join(rastdir_B, "CA3_inh_spikemon_i.txt"))
rast_CA1_E_t = loadtxt(os.path.join(rastdir_B, "CA1_pyCAN_spikemon_t.txt"))
rast_CA1_E_i = loadtxt(os.path.join(rastdir_B, "CA1_pyCAN_spikemon_i.txt"))
rast_CA1_I_t = loadtxt(os.path.join(rastdir_B, "CA1_inh_spikemon_t.txt"))
rast_CA1_I_i = loadtxt(os.path.join(rastdir_B, "CA1_inh_spikemon_i.txt"))

# Area sizes
# N_EC_E = round(max(rast_EC_E_i),-2)
# N_EC_I = round(max(rast_EC_I_i),-2)
# N_DG_E = round(max(rast_DG_E_i),-2)
# N_DG_I = round(max(rast_DG_I_i),-2)
# N_CA3_E = round(max(rast_CA3_E_i),-2)
# N_CA3_I = round(max(rast_CA3_I_i),-2)
# N_CA1_E = round(max(rast_CA1_E_i),-3)
# N_CA1_I = round(max(rast_CA1_I_i),-2)

# print("N_EC: ", N_EC_E, N_EC_I)
# print("N_DG: ", N_DG_E, N_DG_I)
# print("N_CA3: ", N_CA3_E, N_CA3_I)
# print("N_CA1: ", N_CA1_E, N_CA1_I)

# Mean FRs
FR_EC_E_mean = ((len(rast_EC_E_t)/(max(rast_EC_E_t)/1000-min(rast_EC_E_t)/1000))/N_EC_E)
FR_EC_I_mean = ((len(rast_EC_I_t)/(max(rast_EC_I_t)/1000-min(rast_EC_I_t)/1000))/N_EC_I)
FR_DG_E_mean = ((len(rast_DG_E_t)/(max(rast_DG_E_t)/1000-min(rast_DG_E_t)/1000))/N_DG_E)
FR_DG_I_mean = ((len(rast_DG_I_t)/(max(rast_DG_I_t)/1000-min(rast_DG_I_t)/1000))/N_DG_I)
FR_CA3_E_mean = ((len(rast_CA3_E_t)/(max(rast_CA3_E_t)/1000-min(rast_CA3_E_t)/1000))/N_CA3_E)
FR_CA3_I_mean = ((len(rast_CA3_I_t)/(max(rast_CA3_I_t)/1000-min(rast_CA3_I_t)/1000))/N_CA3_I)
FR_CA1_E_mean = ((len(rast_CA1_E_t)/(max(rast_CA3_E_t)/1000-min(rast_CA3_E_t)/1000))/N_CA1_E)
FR_CA1_I_mean = ((len(rast_CA1_I_t)/(max(rast_CA1_I_t)/1000-min(rast_CA1_I_t)/1000))/N_CA1_I)

# Print results
print("Mean FRs - Mine w/ or. pos.")
print("EC:", "\t", round(FR_EC_E_mean,4), "\t", round(FR_EC_I_mean,4))
# print("EC E, tmin:", min(rast_EC_E_t), "tmax:", max(rast_EC_E_t))
# print("EC I, tmin:", min(rast_EC_I_t), "tmax:", max(rast_EC_I_t))
print("DG:", "\t", round(FR_DG_E_mean,4), "\t", round(FR_DG_I_mean,4))
print("CA3:", "\t", round(FR_CA3_E_mean,4), "\t", round(FR_CA3_I_mean,4))
print("CA1:", "\t", round(FR_CA1_E_mean,4), "\t", round(FR_CA1_I_mean,4))
print()

# Amelie's positions
# +=================================+
# Load the rasters
rast_EC_E_t = loadtxt(os.path.join(rastdir_C, "EC_pyCAN_spikemon_t.txt"))
rast_EC_E_i = loadtxt(os.path.join(rastdir_C, "EC_pyCAN_spikemon_i.txt"))
rast_EC_I_t = loadtxt(os.path.join(rastdir_C, "EC_inh_spikemon_t.txt"))
rast_EC_I_i = loadtxt(os.path.join(rastdir_C, "EC_inh_spikemon_i.txt"))
rast_DG_E_t = loadtxt(os.path.join(rastdir_C, "DG_py_spikemon_t.txt"))
rast_DG_E_i = loadtxt(os.path.join(rastdir_C, "DG_py_spikemon_i.txt"))
rast_DG_I_t = loadtxt(os.path.join(rastdir_C, "DG_inh_spikemon_t.txt"))
rast_DG_I_i = loadtxt(os.path.join(rastdir_C, "DG_inh_spikemon_i.txt"))
rast_CA3_E_t = loadtxt(os.path.join(rastdir_C, "CA3_pyCAN_spikemon_t.txt"))
rast_CA3_E_i = loadtxt(os.path.join(rastdir_C, "CA3_pyCAN_spikemon_i.txt"))
rast_CA3_I_t = loadtxt(os.path.join(rastdir_C, "CA3_inh_spikemon_t.txt"))
rast_CA3_I_i = loadtxt(os.path.join(rastdir_C, "CA3_inh_spikemon_i.txt"))
rast_CA1_E_t = loadtxt(os.path.join(rastdir_C, "CA1_pyCAN_spikemon_t.txt"))
rast_CA1_E_i = loadtxt(os.path.join(rastdir_C, "CA1_pyCAN_spikemon_i.txt"))
rast_CA1_I_t = loadtxt(os.path.join(rastdir_C, "CA1_inh_spikemon_t.txt"))
rast_CA1_I_i = loadtxt(os.path.join(rastdir_C, "CA1_inh_spikemon_i.txt"))

# Area sizes
# N_EC_E = round(max(rast_EC_E_i),-2)
# N_EC_I = round(max(rast_EC_I_i),-2)
# N_DG_E = round(max(rast_DG_E_i),-2)
# N_DG_I = round(max(rast_DG_I_i),-2)
# N_CA3_E = round(max(rast_CA3_E_i),-2)
# N_CA3_I = round(max(rast_CA3_I_i),-2)
# N_CA1_E = round(max(rast_CA1_E_i),-3)
# N_CA1_I = round(max(rast_CA1_I_i),-2)

# print("N_EC: ", N_EC_E, N_EC_I)
# print("N_DG: ", N_DG_E, N_DG_I)
# print("N_CA3: ", N_CA3_E, N_CA3_I)
# print("N_CA1: ", N_CA1_E, N_CA1_I)

# Mean FRs
FR_EC_E_mean = ((len(rast_EC_E_t)/(max(rast_EC_E_t)/1000-min(rast_EC_E_t)/1000))/N_EC_E)
FR_EC_I_mean = ((len(rast_EC_I_t)/(max(rast_EC_I_t)/1000-min(rast_EC_I_t)/1000))/N_EC_I)
FR_DG_E_mean = ((len(rast_DG_E_t)/(max(rast_DG_E_t)/1000-min(rast_DG_E_t)/1000))/N_DG_E)
FR_DG_I_mean = ((len(rast_DG_I_t)/(max(rast_DG_I_t)/1000-min(rast_DG_I_t)/1000))/N_DG_I)
FR_CA3_E_mean = ((len(rast_CA3_E_t)/(max(rast_CA3_E_t)/1000-min(rast_CA3_E_t)/1000))/N_CA3_E)
FR_CA3_I_mean = ((len(rast_CA3_I_t)/(max(rast_CA3_I_t)/1000-min(rast_CA3_I_t)/1000))/N_CA3_I)
FR_CA1_E_mean = ((len(rast_CA1_E_t)/(max(rast_CA3_E_t)/1000-min(rast_CA3_E_t)/1000))/N_CA1_E)
FR_CA1_I_mean = ((len(rast_CA1_I_t)/(max(rast_CA1_I_t)/1000-min(rast_CA1_I_t)/1000))/N_CA1_I)

# Print results
print("Mean FRs - Mine w/ or. pos. and eqs.")
print("EC:", "\t", round(FR_EC_E_mean,4), "\t", round(FR_EC_I_mean,4))
print("DG:", "\t", round(FR_DG_E_mean,4), "\t", round(FR_DG_I_mean,4))
print("CA3:", "\t", round(FR_CA3_E_mean,4), "\t", round(FR_CA3_I_mean,4))
print("CA1:", "\t", round(FR_CA1_E_mean,4), "\t", round(FR_CA1_I_mean,4))
