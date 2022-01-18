from .utils import *

##
def kmeans_sklearn(input_data, target_data=None, **kwargs):
  clf = KMeans(**kwargs)
  clf.fit(input_data)
  return {'loss': clf.inertia_}

def kmeans_sklearn_n10_iter10(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, n_clusters=10, max_iter=10)

def kmeans_sklearn_n10_iter10e4(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, n_clusters=10, max_iter=10**4)

def kmeans_sklearn_n10_iter10e5(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, n_clusters=10, max_iter=10**5)

def kmeans_sklearn_n10_iter10e6(input_data, target_data=None):
  return kmeans_sklearn(input_data, target_data, n_clusters=10, max_iter=10**6)
##
