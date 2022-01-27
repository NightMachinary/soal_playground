#!/usr/bin/env python

import sys
import os
import gc

from soalpy.utils import *
from soalpy.runners import *
from soalpy.kmeans_runners import *

from pynight.common_iterable import get_or_none
from pynight.common_debugging import debug_p
from pynight.common_files import dir_ensure

import dask_ml
import dask_ml.datasets
import dask.array as da
##
def nop(*args, **kwargs):
    return None
nop_float64 = nop
##
def save(X, y, **kwargs):
    #: * https://numpy.org/doc/stable/reference/generated/numpy.save.html
    ##
    assert save_dir

    dir_ensure(f"{save_dir}/")

    if isinstance(X, da.Array):
        X.to_zarr(f"{save_dir}/X_zarr", overwrite=True)
        y.to_zarr(f"{save_dir}/y_zarr", overwrite=True)
    else:
        X_path = f"{save_dir}/X.npy"
        np.save(X_path, X, allow_pickle=False)
        np.save(f"{save_dir}/y.npy", y, allow_pickle=False)

    return None
##
def load_np(load_dir):
    #: * https://numpy.org/doc/stable/reference/generated/numpy.memmap.html#numpy.memmap
    # * https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ##
    mmap_mode = "r" #: Open existing file for reading only.
    X = np.load(f"{load_dir}/X.npy", allow_pickle=False, mmap_mode=mmap_mode)
    y = np.load(f"{load_dir}/y.npy", allow_pickle=False, mmap_mode=mmap_mode)

    return X, y


def load_zarr(load_dir):
    X = da.from_zarr(f"{load_dir}/X_zarr")
    y = da.from_zarr(f"{load_dir}/y_zarr")

    return X, y


def blobs(mode='sk'):
    if load_dir:
        print(f"Loading the data from: {load_dir}", file=sys.stderr)
        if mode == 'sk':
            return load_np(load_dir)
        elif mode == 'dask':
            return load_zarr(load_dir)
    else:
        ## @input
        n_samples = int(get_or_none(sys.argv, 3) or 10**4)
        n_features = int(get_or_none(sys.argv, 4) or 100)
        centers = int(get_or_none(sys.argv, 5) or 10)
        ##
        blobs_opts = {
            "n_samples": n_samples,
            "n_features": n_features,
            "centers": centers,
            "random_state": random_state
        }
        if debug_p:
            print(f"algo_name: {algo_name}", file=sys.stderr)
            print(blobs_opts, file=sys.stderr)

        if mode == 'sk':
            X, y = datasets.make_blobs(**blobs_opts)
        elif mode == 'dask':
            X, y = dask_ml.datasets.make_blobs(
                chunks=(10**4, 10**4)),
                **blobs_opts,
            )

    if algo_name == 'nop_float64':
        print("skipped converting the data to float32", file=sys.stderr)
    else:
        X = X.astype(np.float32, copy=False) #: =copy=False= most probably does not work due to the incompatible dtype.
        y = y.astype(np.float32, copy=False)

    return X, y

def blobs_sk(*args, **kwargs):
    return blobs(*args, **kwargs, mode='sk')

def blobs_dask(*args, **kwargs):
    return blobs(*args, **kwargs, mode='dask')
##
g = globals()
## @input
assert len(sys.argv) >= 3

algo_name = sys.argv[1]
algo = g[algo_name]

dataset_name = sys.argv[2]
dataset_get = g[dataset_name]

load_dir = get_or_none(os.environ, "run_one_load_dir")
save_dir = get_or_none(os.environ, "run_one_save_dir")

if algo_name == 'save' and os.path.exists(save_dir):
    #: Don't recreate the datasets if they already exist.
    print("skipped saving the dataset", file=sys.stderr)
    sys.exit(0)

random_state = int(get_or_none(os.environ, 'random_state') or 42)
##
X, y = dataset_get()
gc.collect()

res = algo(X, y)

if res is not None:
    print(
        f"{res['loss']},{res['homogeneity_score']},{res['completeness_score']},{res['adjusted_rand_score']}"
    )
