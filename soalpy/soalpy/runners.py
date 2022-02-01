from .utils import *
from pynight.common_redirections import stdout_redirected
from contextlib import ExitStack

from sklearn.cluster import MiniBatchKMeans, KMeans

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
    **kwargs
):
    #: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    #: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
    ##
    input_data = dataset['input_data']
    target_data = get_or_none(dataset, 'target_data')
    n_clusters = dataset['n_clusters']

    res = dict()
    clf = None
    gpu_p = False

    hdbscan_p = "HDBSCAN" in mode
    kmeans_p = "kmeans" in mode.lower()
    spectral_p = "spectral" in mode.lower()

    if kmeans_p:
        kwargs['n_clusters'] = n_clusters

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

        gpu_p = True
        clf = cuHDBSCAN(min_cluster_size=10 ** 2, verbose=0, **kwargs,)
    elif mode == "spectral_dask":
        clf = dask_ml.cluster.SpectralClustering(
            n_jobs=-1,
            **kwargs,
            )

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
    elif spectral_p:
        res["loss"] = 0

    return res
###
def hdbscan_cuml(dataset):
    return run(dataset, mode="cuHDBSCAN")
###
