# -----------------------------------------------------------------------------
# Memory Stimulation and Phase-Amplitude Coupling
# Copyright 2021 Nikolaos Vardlakis & Nicolas P. Rougier
# Released under the BSD 3-clauses license
# -----------------------------------------------------------------------------
import os
import json
import time
import subprocess

# Constants


# Default parameters
_data = {
    "seed"                  : 42,       # Reproducibility

    # additive noise terms
    "noise": {
        "py": {
            "sigma"         : 0.
        },
        "inh": {
            "sigma"         : 0.
        }
    },

    # areas, tables 3.1-3.3, pages 45-48, Aussel
    "areas": {
        "EC"    : {
            "E" : {
                "N" : 10e3,
                "type" : "PyCAN"
            },
            "I" : {
                "N" : 1e3,
                "type" : "Inh"
            }
        },
        "DG"    : {
            "E" : {
                "N" : 10e3,
                "type" : "Py"
            },
            "I" : {
                "N" : 0.1e3,
                "type" : "Inh"
            }
        },
        "CA3"   : {
            "E" : {
                "N" : 1e3,
                "type" : "PyCAN"
            },
                "I" : {
                "N" : 0.1e3,
                "type" : "Inh"
            }
        },
        "CA1"   : {
            "E" : {
                "N" : 10e3,
                "type" : "PyCAN"
            },
            "I" : {
                "N" : 1e3,
                "type" : "Inh"
            }
        }
    },

    # Kuramoto oscillator parameters
    "Kuramoto" : {
        "N" : 50,
        "f0" : 4.,
        "sigma" : 0.5,  # normal std
        "kN" : 5
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
            "p_mono" : 0.2 # mono: [EC->CA3, EC->CA1]
        }
    },

    # synapses
    "synapses" : {
        "gmax_e" : 600., # pSiemens
        "gmax_i" : 60.
    },

    # stimulation parameters
    "stimulation" : {},

    # simulation parameters
    "simulation" : {
        "duration" : 1e3 # ms
    },

    # git stuff
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

    parser = argparse.ArgumentParser(
        description='Generate parameters file using JSON format')
    parser.add_argument('parameters_file',
                        default='default',
                        type=str, nargs='?',
                        help='Parameters file (json format)')
    args = parser.parse_args()

    filename = "./configs/{0}.json".format(args.parameters_file)

    print('Saving file "{0}"'.format(filename))
    save(filename, _data)

    print('..:: Unit Testing ::..')
    print('----------------------------------')
    data = load(filename)
    dump(data)
    print('----------------------------------')

    locals().update(data)
    print('Saving file "{0}"'.format(filename))
    save(filename)
