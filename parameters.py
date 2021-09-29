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
            "sigma"         : 0
        },
        "inh": {
            "sigma"         : 0
        }
    },

    # cells
    "celltypes": {
        "py": {
            "A"             : 29000,
            "gl"            : 0.01,
            "El"            : -70,
            "gK"            : 5,
            "EK"            : -100,
            "gNa"           : 50,
            "ENa"           : 50,
            "gM"            : 90,
            "EM"            : -100,
            "gCa"           : 0.1,
            "ECa"           : 120,
            "gCAN"          : 0.0005,
            "ECAN"          : -20
        },
        "inh": {
            "A"             : 14000,
            "gl"            : 0.1,
            "El"            : -90,
            "gK"            : 9,
            "EK"            : -65,
            "gNa"           : 35,
            "ENa"           : 55
        }
    },

    # simulation parameters

    # stimulation parameters

    # git stuff
    "git":{
        "timestamp"         : None,
        "git_branch"        : None,
        "git_hash"          : None,
        "git_short_hash"    : None
    }
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
    _data["git_short_hash"] = get_git_revision_short_hash()
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
                        default="default",
                        type=str, nargs="?",
                        help='Parameters file (json format)')
    args = parser.parse_args()

    filename = "./configs/{0}.json".format(args.parameters_file)

    print('Saving file "{0}"'.format(filename))
    save(filename, _data)

    print('..::Parameters::..')
    print('----------------------------------')
    data = load(filename)
    dump(data)
    print('----------------------------------')

    locals().update(data)
    print('Saving file "{0}"'.format(filename))
    save(filename)
