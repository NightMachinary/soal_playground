#!/usr/bin/env python3

from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans, KMeans

import dask_ml
import dask_ml.cluster
##
def spectral_dask_est1(dataset):
    input_data = dataset['input_data']
    target_data = dataset['target_data']
    n_clusters = dataset['n_clusters']

    batch_size = 2 ** 10
    estimator = MiniBatchKMeans(
        batch_size=batch_size,
        n_clusters=n_clusters,
        max_iter=10 ** 4,
    )

    clf = dask_ml.cluster.SpectralClustering(
        n_jobs=-1,
        assign_labels=estimator,
        )

    clf.fit(input_data)
##
centers = 10
blobs_opts = {
    "n_samples": 10**4,
    "n_features": 10**4,
    "centers": centers,
    "random_state": 0,
}

X, y = datasets.make_blobs(**blobs_opts)
dataset = {
    'input_data': X,
    'target_data': y,
    'n_clusters': centers,
}

spectral_dask_est1(dataset)
