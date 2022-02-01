from .utils import *
from .runners import *

kmeans_sklearn = run
###
def kmeans_sklearn_full(*args, **kwargs):
  return kmeans_sklearn(*args, algorithm='full', **kwargs)
##
def kmeans_sklearn_iter10(dataset):
  return kmeans_sklearn(dataset, max_iter=10)

def kmeans_sklearn_iter10e4(dataset):
  return kmeans_sklearn(dataset, max_iter=10**4)

def kmeans_sklearn_iter10e5(dataset):
  return kmeans_sklearn(dataset, max_iter=10**5)

def kmeans_sklearn_iter10e6(dataset):
  return kmeans_sklearn(dataset, max_iter=10**6)
##
def kmeans_sklearn_full_iter10e4(dataset):
  return kmeans_sklearn_full(dataset, max_iter=10**4)
##
def kmeans_mb2e13_sklearn_iter10e4(dataset):
  return kmeans_sklearn(dataset, mode='MiniBatchKMeans', batch_size=2**13, max_iter=10**4)

def kmeans_mb2e10_sklearn_iter10e4(dataset):
  return kmeans_sklearn(dataset, mode='MiniBatchKMeans', batch_size=2**10, max_iter=10**4)

def kmeans_mb2e10_sklearn_n2_iter10e4(dataset):
  return kmeans_sklearn(dataset, mode='MiniBatchKMeans', batch_size=2**10, n_clusters=2, max_iter=10**4)

def kmeans_mb2e7_sklearn_iter10e4(dataset):
  return kmeans_sklearn(dataset, mode='MiniBatchKMeans', batch_size=2**7, max_iter=10**4)

def kmeans_mb2e7_sklearn_iter10e4_no_metrics(dataset):
  return kmeans_sklearn(dataset, mode='MiniBatchKMeans', batch_size=2**7, max_iter=10**4, no_metrics=True)
##
def kmeans_mb2e10_sklearn_iter10e4_dask(dataset):
  return kmeans_sklearn(dataset, mode='MiniBatchKMeans', batch_size=2**10, max_iter=10**4, dask_p=True, dask_incremental=True)


def kmeans_mb2e10_sklearn_iter10e4_dask_no_metrics(dataset):
  return kmeans_sklearn(dataset, mode='MiniBatchKMeans', batch_size=2**10, max_iter=10**4, dask_p=True, dask_incremental=True, no_metrics=True)


def kmeans_mb2e7_sklearn_iter10e4_dask(dataset):
  return kmeans_sklearn(dataset, mode='MiniBatchKMeans', batch_size=2**7, max_iter=10**4, dask_p=True, dask_incremental=True)
##
def kmeans_dask_iter10e4(dataset):
  return kmeans_sklearn(dataset, mode='kmeans_dask', max_iter=10**4, dask_p=True)
##
def kmeans_b3_sklearn_iter10e4(dataset):
  return kmeans_sklearn(dataset, mode='BKMeans', breathing_depth=3, max_iter=10**4)
##
def kmeans_cuml_iter10e4(dataset):
  return kmeans_sklearn(dataset, mode='cuKMeans', max_iter=10**4)
###
