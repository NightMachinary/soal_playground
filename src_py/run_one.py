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

g = globals()
## input arguments
assert len(sys.argv) >= 2

algo_name = sys.argv[1]
algo = g[algo_name]
n_samples = int(get_or_none(sys.argv, 2) or 10**4)
n_features = int(get_or_none(sys.argv, 3) or 100)
centers = int(get_or_none(sys.argv, 4) or 10)

random_state = int(get_or_none(os.environ, 'random_state') or 42)
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

X, y = datasets.make_blobs(**blobs_opts)
X = X.astype(np.float32, copy=False) #: =copy=False= most probably does not work due to the incompatible dtype.
y = y.astype(np.float32, copy=False)
gc.collect()

res = algo(X, y)

if res is not None:
    print(
        f"{res['loss']},{res['homogeneity_score']},{res['completeness_score']},{res['adjusted_rand_score']}"
    )
