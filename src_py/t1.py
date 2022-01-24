#!/usr/bin/env python3

import numpy
np = numpy
import sys

import dask_ml
import dask_ml.cluster
from dask_ml.datasets import make_blobs as dask_make_blobs
##
X, y = dask_make_blobs(
    n_samples=10**3,
    n_features=2,
    # n_samples=10**6,
    # n_features=10**4,
    centers=10,
    chunks=(10**4, 10**4),
    # chunks=(10**4, 10**4),
    )
X = X.astype(np.float32)
y = y.astype(np.float32)
##
import cupy

X = X.map_blocks(cupy.asarray)
##
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask

cluster = LocalCUDACluster(dashboard_address=None)
client = Client(cluster)
##
from cuml.dask.cluster import KMeans as cuKMeans

km_cu = cuKMeans(n_clusters=10)
km_cu.fit(X)
