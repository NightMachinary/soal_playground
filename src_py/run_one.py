#!/usr/bin/env python

import sys
import os
import gc
from soalpy.utils import *
from soalpy.kmeans_runners import *
from pynight.common_iterable import get_or_none
from pynight.common_debugging import debug_p

def nop(*args, **kwargs):
    return None
nop_float64 = nop

def save(X, y, **kwargs):
    #: * https://numpy.org/doc/stable/reference/generated/numpy.save.html
    ##
    save_dir = get_or_none(os.environ, "run_one_save_dir") #: @input
    assert save_dir

    np.save(f"{save_dir}/X.npy", X, allow_pickle=False)
    np.save(f"{save_dir}/y.npy", y, allow_pickle=False)
    return None

g = globals()
## @input
assert len(sys.argv) >= 2

algo_name = sys.argv[1]
algo = g[algo_name]

load_dir = get_or_none(os.environ, "run_one_load_dir")
n_samples = int(get_or_none(sys.argv, 2) or 10**4)
n_features = int(get_or_none(sys.argv, 3) or 100)
centers = int(get_or_none(sys.argv, 4) or 10)

random_state = int(get_or_none(os.environ, 'random_state') or 42)
##
if load_dir:
    #: * https://numpy.org/doc/stable/reference/generated/numpy.memmap.html#numpy.memmap
    # * https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ##
    print("Loading the data from: {load_dir}", file=sys.stderr)
    mmap_mode = "r" #: Open existing file for reading only.
    X = np.load(f"{load_dir}/X.npy", allow_pickle=False, mmap_mode=mmap_mode)
    y = np.load(f"{load_dir}/y.npy", allow_pickle=False, mmap_mode=mmap_mode)
else:
    blobs_opts = {
        "n_samples": n_samples,
        "n_features": n_features,
        "centers": centers,
        "random_state": random_state
    }
    if debug_p:
        print(f"algo_name: {algo_name}", file=sys.stderr)
        print(blobs_opts, file=sys.stderr)

    X, y = datasets.make_blobs(**blobs_opts)
    if algo_name == 'nop_float64':
        print("skipped converting the data to float32", file=sys.stderr)
    else:
        X = X.astype(np.float32, copy=False) #: =copy=False= most probably does not work due to the incompatible dtype.
        y = y.astype(np.float32, copy=False)

gc.collect()

res = algo(X, y)

if res is not None:
    print(
        f"{res['loss']},{res['homogeneity_score']},{res['completeness_score']},{res['adjusted_rand_score']}"
    )
