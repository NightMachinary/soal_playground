from .utils import *
from icecream import ic
from pynight.common_redirections import stdout_redirected
from contextlib import ExitStack

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

from hdbscan import HDBSCAN

#: https://ml.dask.org/incremental.html
from dask_ml.wrappers import Incremental

try:
    from bkmeans import BKMeans
except ImportError:
    print("bkmeans not installed", file=sys.stderr)

# import numba as nb

try:
    import cudf
    import cuml

    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import HDBSCAN as cuHDBSCAN
except ImportError:
    print("RAPIDS not installed", file=sys.stderr)

def run(
    dataset,
    mode="KMeans",
    batch_size=2 ** 10,
    breathing_depth=3,
    no_metrics=False,
    dask_p=False,
    dask_incremental=False,
    distance_mat_p=True,
    **kwargs
):
    #: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    #: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
    ##
    input_data = dataset['input_data']
    target_data = get_or_none(dataset, 'target_data')
    n_clusters = dataset['n_clusters']
    input_is_distance = dataset['input_is_distance']

    min_cluster_size = 10 ** 1

    res = dict()
    clf = None
    gpu_p = False

    hdbscan_p = "HDBSCAN" in mode
    kmeans_p = "kmeans" in mode.lower()
    gmm_p = "gmm" in mode.lower()
    spectral_p = "spectral" in mode.lower()

    if kmeans_p:
        kwargs['n_clusters'] = n_clusters

    if hdbscan_p:
        kwargs.setdefault('min_cluster_size', min_cluster_size)

        metric = kwargs.get('metric', 'euclidean')
        if distance_mat_p and input_is_distance:
            if 'metric' in kwargs:
                print(f"WARNING: metric forcefully switched from {metric} to precomputed.", file=sys.stderr)
            else:
                print(f"INFO: metric switched to precomputed.", file=sys.stderr)

            metric = 'precomputed'

        kwargs['metric'] = metric

    if mode == "KMeans":
        clf = KMeans(**kwargs)
    elif mode == "kmeans_dask":
        clf = dask_ml.cluster.KMeans(
            n_jobs=-1,
            **kwargs,
        )
    elif mode == "BKMeans":
        clf = BKMeans(
            #: The parameter m (breathing depth) can be used to generate faster ( 1 < m < 5) or better (m>5) solutions.
            m=breathing_depth,
            **kwargs,
        )
    elif mode == "cuKMeans":
        gpu_p = True
        clf = cuKMeans(**kwargs)
    elif mode == "MiniBatchKMeans":
        clf = MiniBatchKMeans(
            batch_size=batch_size,
            compute_labels=True,
            # max_no_improvement=10,
            **kwargs,
        )
    elif mode == "cuHDBSCAN":
        #: https://docs.rapids.ai/api/cuml/stable/api.html#hdbscan
        #: =algorithm= not supported (has a single algorithm)

        gpu_p = True
        clf = cuHDBSCAN(verbose=0, **kwargs,)
    elif mode == "HDBSCAN":
        #: https://github.com/scikit-learn-contrib/hdbscan
        #: http://hdbscan.readthedocs.io/en/latest/
        #: https://hdbscan.readthedocs.io/en/latest/api.html

        clf = HDBSCAN(**kwargs,)
    elif mode == "spectral_dask":
        clf = dask_ml.cluster.SpectralClustering(
            n_jobs=-1,
            **kwargs,
            )
    elif mode == "spectral_sklearn":
        affinity = kwargs.get('affinity', 'rbf')
        if distance_mat_p and input_is_distance:
            affinity = 'precomputed_nearest_neighbors'

            print(f"INFO: affinity switched to {affinity}.", file=sys.stderr)

        kwargs['affinity'] = affinity

        clf = SpectralClustering(
            n_jobs=-1,
            **kwargs,
            )
    elif mode == "GMM":
        clf = GaussianMixture(
            n_components=n_clusters,
            # init_params='k-means++', #: needs at least sklearn v1.1
            **kwargs,
        )

    # ic(type(input_data))
    if type(input_data) != np.ndarray and hasattr(input_data, 'to_numpy'): #: e.g., xarray.core.dataarray.DataArray
        input_data = input_data.to_numpy()
        # ic(type(input_data))

    if gpu_p:
        ##
        # input_data = nb.cuda.to_device(input_data)
        ##
        input_data = cudf.DataFrame(input_data)
        ##

    if dask_incremental:
        clf = Incremental(clf)

    with ExitStack() as exit_stack:
        if mode == "cuHDBSCAN":
            #: cuHDBSCAN writes to stdout, hence the redirection.
            exit_stack.enter_context(stdout_redirected(sys.stderr))

        if target_data is not None:
            if no_metrics:
                clf.fit(input_data)

                res["homogeneity_score"] = 0
                res["completeness_score"] = 0
                res["adjusted_rand_score"] = 0
            else:
                if hasattr(clf, 'fit_predict'):
                    preds = clf.fit_predict(input_data)
                else:
                    clf.fit(input_data)
                    preds = clf.predict(input_data)

                if gpu_p:
                    preds = preds.to_numpy()
                ##
                #: It's possible that computing these metrics can change the max memory usage.
                if dask_p:
                    res["homogeneity_score"] = homogeneity_score_dask(target_data, preds)
                    res["completeness_score"] = completeness_score_dask(
                        target_data, preds
                    )
                    res["adjusted_rand_score"] = adjusted_rand_score_dask(
                        target_data, preds
                    )
                else:
                    res["homogeneity_score"] = metrics.homogeneity_score(target_data, preds)
                    res["completeness_score"] = metrics.completeness_score(
                        target_data, preds
                    )

                    #: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
                    res["adjusted_rand_score"] = metrics.adjusted_rand_score(
                        target_data, preds
                    )
        else:
            clf.fit(input_data)

    if kmeans_p:
        res["loss"] = clf.inertia_
    elif hdbscan_p:
        probs = clf.probabilities_
        if gpu_p:
            probs = probs.to_numpy()

        res["loss"] = -np.mean(probs)
    elif hasattr(clf, 'score'): #: gmm_p
        res["loss"] = -clf.score(input_data)
    else: #: spectral_p
        res["loss"] = 0

    return res
###
