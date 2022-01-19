from .utils import *

###
def kmeans_sklearn(input_data, target_data=None, mode='KMeans', **kwargs):
  #: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
  #: * https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
  ##
  res = dict()
  clf = None
  if mode == 'KMeans':
    clf = KMeans(**kwargs)
  elif mode == 'MiniBatchKMeans':
    clf = MiniBatchKMeans(
      batch_size=2**10,
      compute_labels=True,
      # max_no_improvement=10,
      **kwargs
    )

  if (target_data is not None):
    preds = clf.fit_predict(input_data)
    res['homogeneity_score'] = metrics.homogeneity_score(target_data, preds)
    res['completeness_score'] = metrics.completeness_score(target_data, preds)
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
def kmeans_mb_sklearn_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, mode='MiniBatchKMeans', n_clusters=10, max_iter=10**4)
###
