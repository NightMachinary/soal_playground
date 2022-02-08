#!/usr/bin/env python

import sys
import os
import gc

from soalpy.utils import *
from soalpy.runners import *
from soalpy.kmeans_runners import *
from soalpy.spectral_runners import *
from soalpy.hdbscan_runners import *

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
def save(dataset, **kwargs):
    #: * https://numpy.org/doc/stable/reference/generated/numpy.save.html
    ##
    X = dataset['input_data']
    y = get_or_none(dataset, 'target_data')
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
    ## @input
    n_samples = int(get_or_none(sys.argv, 3) or 10**4)
    n_features = int(get_or_none(sys.argv, 4) or 100)
    centers = int(get_or_none(sys.argv, 5) or 10)
    ##
    if load_dir:
        print(f"Loading the data from: {load_dir}", file=sys.stderr)
        if mode == 'sk':
            X, y = load_np(load_dir)
        elif mode == 'dask':
            X, y = load_zarr(load_dir)
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

        if mode == 'sk':
            X, y = datasets.make_blobs(**blobs_opts)
        elif mode == 'dask':
            X, y = dask_ml.datasets.make_blobs(
                chunks=(10**4, 10**4),
                **blobs_opts,
            )

        if algo_name == 'nop_float64':
            print("skipped converting the data to float32", file=sys.stderr)
        else:
            X = X.astype(np.float32, copy=False) #: =copy=False= most probably does not work due to the incompatible dtype.
            y = y.astype(np.float32, copy=False)

    dataset = {
        'input_data': X,
        'input_is_distance': False,
        'target_data': y,
        'n_clusters': centers,
    }
    return dataset


def blobs_sk(*args, **kwargs):
    return blobs(*args, **kwargs, mode='sk')

def blobs_dask(*args, **kwargs):
    return blobs(*args, **kwargs, mode='dask')
##
import rdata

def fcps_gen(ds_name, n_clusters=None):
    parsed = rdata.parser.parse_file(f'{fcps_dir}/data/{ds_name}.rda')
    converted = rdata.conversion.convert(parsed)
    distance_mat = get_or_none(converted[ds_name], 'DistanceMatrix')
    X = get_or_none(converted[ds_name], 'Data')
    X_is_distance = False
    if X is None:
        X_is_distance = True
        X = distance_mat

    y = converted[ds_name]['Cls']
    if n_clusters is None:
        n_clusters = len(np.unique(y))

        print(f"n_clusters inferred from target_data to be {n_clusters}", file=sys.stderr)

    ## shuffles X and y together:
    indices = np.arange(len(y))
    np.random.shuffle(indices) #: @inplace
    X = X[indices]
    y = y[indices]
    ##

    dataset = {
        'input_data': X,
        'input_is_distance': X_is_distance,
        'distance_mat': distance_mat,
        'target_data': y,
        'n_clusters': n_clusters,
    }
    return dataset


def fcps_atom():
    return fcps_gen('Atom')


def fcps_chainlink():
    return fcps_gen('Chainlink')


def fcps_dietary_survey_IBS():
    return fcps_gen('dietary_survey_IBS')


def fcps_engy_time():
    return fcps_gen('EngyTime')


def fcps_golf_ball():
    return fcps_gen('GolfBall')


def fcps_hepta():
    return fcps_gen('Hepta')


def fcps_leukemia():
    return fcps_gen('Leukemia', 5)


def fcps_lsun3D():
    return fcps_gen('Lsun3D')


def fcps_target():
    return fcps_gen('Target')

def fcps_tetra():
    return fcps_gen('Tetra')


def fcps_twodiamonds():
    return fcps_gen('TwoDiamonds', 2)


def fcps_wing_nut():
    return fcps_gen('WingNut')
##
g = globals()
## @input
assert len(sys.argv) >= 3

algo_name = sys.argv[1]
algo = g[algo_name]

dataset_name = sys.argv[2]
if debug_p:
    print(f"dataset_name={dataset_name}", file=sys.stderr)

dataset_get = g[dataset_name]

load_dir = get_or_none(os.environ, "run_one_load_dir")
save_dir = get_or_none(os.environ, "run_one_save_dir")

fcps_dir = get_or_none(os.environ, 'fcps_dir') or f'{os.environ["HOME"]}/Base/_Code/misc/FCPS'

if algo_name == 'save' and os.path.exists(save_dir):
    #: Don't recreate the datasets if they already exist.
    print("skipped saving the dataset", file=sys.stderr)
    sys.exit(0)

random_state = int(get_or_none(os.environ, 'random_state') or 42)
##
dataset = dataset_get()
gc.collect()

res = algo(dataset)

if res is not None:
    print(
        f"{res['loss']},{res['homogeneity_score']},{res['completeness_score']},{res['adjusted_rand_score']}"
    )
