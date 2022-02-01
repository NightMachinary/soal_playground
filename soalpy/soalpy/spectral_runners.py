#: * https://ml.dask.org/modules/generated/dask_ml.cluster.SpectralClustering.html
#:
#: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
##
from .utils import *
from .runners import *
##
def spectral_dask_n10_est1(input_data, target_data=None):
    batch_size = 2 ** 10
    estimator = MiniBatchKMeans(
        batch_size=batch_size,
        # compute_labels=True,
        # max_no_improvement=10,
        n_clusters=10,
        max_iter=10 ** 4,
    )

    return kmeans_sklearn(
        input_data,
        target_data,
        mode="spectral_dask",
        n_clusters=10,
        assign_labels=estimator,
        dask_p=True,
    )
##
