from .utils import *
from .runners import *
##
def hdbscan_cuml(dataset):
    return run(dataset, mode="cuHDBSCAN")
##
def hdbscan_sklearn_best(dataset, **kwargs):
    return run(dataset, mode="HDBSCAN", algorithm='best', **kwargs,)

def hdbscan_sklearn_best_nodist(*args, **kwargs):
    return hdbscan_sklearn_best(*args, **kwargs, distance_mat_p=False,)
##
