from .utils import *
from .runners import *

kmeans_sklearn = run
###
def kmeans_sklearn_full(*args, **kwargs):
  return kmeans_sklearn(*args, algorithm='full', **kwargs)
##
def kmeans_sklearn_n10_iter10(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, n_clusters=10, max_iter=10)

def kmeans_sklearn_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, n_clusters=10, max_iter=10**4)

def kmeans_sklearn_n10_iter10e5(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, n_clusters=10, max_iter=10**5)

def kmeans_sklearn_n10_iter10e6(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, n_clusters=10, max_iter=10**6)
##
def kmeans_sklearn_full_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn_full(input_data, target_data, n_clusters=10, max_iter=10**4)
##
def kmeans_mb2e13_sklearn_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='MiniBatchKMeans', batch_size=2**13, n_clusters=10, max_iter=10**4)

def kmeans_mb2e10_sklearn_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='MiniBatchKMeans', batch_size=2**10, n_clusters=10, max_iter=10**4)

def kmeans_mb2e7_sklearn_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='MiniBatchKMeans', batch_size=2**7, n_clusters=10, max_iter=10**4)

def kmeans_mb2e7_sklearn_n10_iter10e4_no_metrics(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='MiniBatchKMeans', batch_size=2**7, n_clusters=10, max_iter=10**4, no_metrics=True)
##
def kmeans_mb2e10_sklearn_n10_iter10e4_dask(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='MiniBatchKMeans', batch_size=2**10, n_clusters=10, max_iter=10**4, dask_p=True, dask_incremental=True)

def kmeans_mb2e7_sklearn_n10_iter10e4_dask(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='MiniBatchKMeans', batch_size=2**7, n_clusters=10, max_iter=10**4, dask_p=True, dask_incremental=True)
##
def kmeans_dask_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='kmeans_dask', n_clusters=10, max_iter=10**4, dask_p=True)
##
def kmeans_b3_sklearn_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='BKMeans', breathing_depth=3, n_clusters=10, max_iter=10**4)
##
def kmeans_cuml_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='cuKMeans', n_clusters=10, max_iter=10**4)
###
