from .utils import *

from sklearn.cluster import MiniBatchKMeans, KMeans

from bkmeans import BKMeans

import cudf
import cuml
# import numba as nb
from cuml.cluster import KMeans as cuKMeans
from cuml.cluster import HDBSCAN as cuHDBSCAN

def run(input_data,
                   target_data=None,
                   mode='KMeans',
                   batch_size=2**10,
                   breathing_depth=3,
                   no_metrics=False,
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
  elif mode == 'MiniBatchKMeans':
    clf = MiniBatchKMeans(
      batch_size=batch_size,
      compute_labels=True,
      # max_no_improvement=10,
      **kwargs
    )
  elif mode == 'cuHDBSCAN':
    #: https://docs.rapids.ai/api/cuml/stable/api.html#hdbscan

    gpu_p = True
    clf = cuHDBSCAN(
        min_cluster_size=10**2,
        verbose=0,
        **kwargs,
    )

  if gpu_p:
    ##
    # input_data = nb.cuda.to_device(input_data)
    ##
    input_data = cudf.DataFrame(input_data)
    ##

  stdout_tmp = sys.stdout
  sys.stdout = sys.stderr
  if (target_data is not None):
    preds = clf.fit_predict(input_data)
    if gpu_p:
      preds = preds.to_numpy()

    if no_metrics:
      res['homogeneity_score'] = 0
      res['completeness_score'] = 0
      res['adjusted_rand_score'] = 0
    else:
      #: It's possible that computing these metrics can change the max memory usage.

      res['homogeneity_score'] = metrics.homogeneity_score(target_data, preds)
      res['completeness_score'] = metrics.completeness_score(target_data, preds)

      #: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
      res['adjusted_rand_score'] = metrics.adjusted_rand_score(target_data, preds)
  else:
    clf.fit(input_data)

  sys.stdout = stdout_tmp

  if 'KMeans' in mode:
      res['loss'] = clf.inertia_
  elif 'HDBSCAN' in mode:
      probs = clf.probabilities_
      if gpu_p:
          probs = probs.to_numpy()

      res['loss'] = -np.mean(probs)

  return res

###
def hdbscan_cuml(input_data, target_data=None):
  return run(input_data, target_data, mode='cuHDBSCAN')
###
