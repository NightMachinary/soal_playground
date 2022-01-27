#!/usr/bin/env python3

import numpy
np = numpy

import dask_ml
import dask_ml.cluster
from dask_ml.datasets import make_blobs as dask_make_blobs
##
X, y = dask_make_blobs(
    # n_samples=10**3,
    n_samples=10**6,
    n_features=10**4,
    centers=10,
    # chunks=(10**4, 10**4),
    chunks=(10**3, 10**4),
)
X = X.astype(np.float32)
y = y.astype(np.float32)
##
km = dask_ml.cluster.KMeans(n_clusters=10, init_max_iter=10)
km.fit(X)
##
