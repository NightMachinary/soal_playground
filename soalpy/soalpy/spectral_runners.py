#: * https://ml.dask.org/modules/generated/dask_ml.cluster.SpectralClustering.html
#:
#: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
##
from .utils import *
from .runners import *

from sklearn.cluster import MiniBatchKMeans, KMeans
##
def spectral_dask_est1(dataset):
    n_clusters = dataset['n_clusters']

    batch_size = 2 ** 10
    estimator = MiniBatchKMeans(
        batch_size=batch_size,
        # compute_labels=True,
        # max_no_improvement=10,
        n_clusters=n_clusters,
        max_iter=10 ** 4,
    )

    return run(
        dataset,
        mode="spectral_dask",
        # n_clusters=n_clusters, #: The dimension of the projection subspace.
        assign_labels=estimator,
        dask_p=True,
    )
##
