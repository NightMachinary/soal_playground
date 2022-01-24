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
if True:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import dask

    #: https://docs.dask.org/en/latest/deploying-python.html
    cluster = LocalCUDACluster(dashboard_address=None) #: errors, idk why
    client = Client(cluster)
    ##
    from cuml.dask.cluster import KMeans as cuKMeans
else:
    from cuml.cluster import KMeans as cuKMeans
# Traceback (most recent call last):
#   File "t1.py", line 41, in <module>
#     km_cu.fit(X)
#   File "/usr/local/lib/python3.8/site-packages/cuml/internals/api_decorators.py", line 409, in inner_with_setters
#     return func(*args, **kwargs)
#   File "cuml/cluster/kmeans.pyx", line 340, in cuml.cluster.kmeans.KMeans.fit
#   File "/usr/local/lib/python3.8/contextlib.py", line 75, in inner
#     return func(*args, **kwds)
#   File "/usr/local/lib/python3.8/site-packages/cuml/internals/api_decorators.py", line 360, in inner
#     return func(*args, **kwargs)
#   File "/usr/local/lib/python3.8/site-packages/cuml/common/input_utils.py", line 379, in input_to_cuml_array
#     raise TypeError(msg)
# TypeError: X matrix format <class 'dask.array.core.Array'> not supported
##

km_cu = cuKMeans(n_clusters=10)
km_cu.fit(X)
