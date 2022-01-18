#!/usr/bin/env python

import sys
import os
from soalpy.utils import *
from soalpy.kmeans_runners import *
from pynight.common_iterable import get_or_none
from pynight.common_debugging import debug_p

g = globals()
## input arguments
assert len(sys.argv) >= 2

algo_name = sys.argv[1]
algo = g[algo_name]
n_samples = get_or_none(sys.argv, 2) or 10**4
n_features = get_or_none(sys.argv, 3) or 100
centers = get_or_none(sys.argv, 4) or 10
##
blobs_opts = {
    "n_samples": n_samples,
    "n_features": n_features,
    "centers": centers,
    "random_state": 42
}
if debug_p:
    print(blobs_opts, file=sys.stderr)

X, y = datasets.make_blobs(**blobs_opts)

res = algo(X, y)

print(res['loss'])
