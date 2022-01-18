from .utils import *

def kmeans_sklearn_n10_iter10(input_data, target_data=None):
  clf = KMeans(n_clusters=10, max_iter=10)
  clf.fit(input_data)
  return {'loss': clf.inertia_}
