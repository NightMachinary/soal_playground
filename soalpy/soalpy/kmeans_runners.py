from .utils import *

from sklearn.cluster import MiniBatchKMeans, KMeans


from bkmeans import BKMeans

import cudf
import cuml
# import numba as nb
from cuml.cluster import KMeans as cuKMeans
###
def kmeans_sklearn(input_data,
                   target_data=None,
                   mode='KMeans',
                   batch_size=2**10,
                   breathing_depth=3,
                   **kwargs):
  #: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
  #: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
  ##
  res = dict()
  clf = None
  gpu_p = False
  if mode == 'KMeans':
    clf = KMeans(**kwargs)
  elif mode == 'BKMeans':
    clf = BKMeans(
      #: The parameter m (breathing depth) can be used to generate faster ( 1 < m < 5) or better (m>5) solutions.
      m=breathing_depth,
      **kwargs)
  elif mode == 'cuKMeans':
    gpu_p = True
    clf = cuKMeans(**kwargs)
    ##
    # input_data = nb.cuda.to_device(input_data)
    ##
    input_data = cudf.DataFrame(input_data)
    ##
  elif mode == 'MiniBatchKMeans':
    clf = MiniBatchKMeans(
      batch_size=batch_size,
      compute_labels=True,
      # max_no_improvement=10,
      **kwargs
    )

  if (target_data is not None):
    preds = clf.fit_predict(input_data)
    if gpu_p:
      preds = preds.to_numpy()

    res['homogeneity_score'] = metrics.homogeneity_score(target_data, preds)
    res['completeness_score'] = metrics.completeness_score(target_data, preds)

    #: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    res['adjusted_rand_score'] = metrics.adjusted_rand_score(target_data, preds)
  else:
    clf.fit(input_data)

  res['loss'] = clf.inertia_
  return res

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
##
def kmeans_b3_sklearn_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='BKMeans', breathing_depth=3, n_clusters=10, max_iter=10**4)
##
def kmeans_cuml_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='cuKMeans', n_clusters=10, max_iter=10**4)
###
