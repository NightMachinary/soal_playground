from .utils import *
from .runners import *
##
def gmm_sklearn_cov_full(dataset, **kwargs):
    return run(dataset, mode="GMM", **kwargs,)
##
