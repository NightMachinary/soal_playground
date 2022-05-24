from .utils import *
from .runners import *
##
def gmm_sklearn_best(dataset, **kwargs):
    return run(dataset, mode="GMM", **kwargs,)
##
