import sys
import os
import numpy as np
import pandas
pd = pandas
from time import time, sleep
from functools import partial
import concurrent
import gc
import matplotlib.pyplot as plt

from fastai.torch_core import show_image, show_images
##
from sklearn import datasets

from sklearn.svm import SVC

from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
#: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#: https://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html

from sklearn.preprocessing import MinMaxScaler
#: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
##
import dask
import dask.array as da

import dask_ml
import dask_ml.cluster
##
def dask_map_zip(fn, a, b, aggregator=np.mean):
  def h_0(a_blocks, b_blocks):
      result = np.zeros(len(a_blocks))

      i = 0
      for a, b in zip(a_blocks, b_blocks):
          result[i] = fn(a, b)
          i += 1

      return aggregator(result)

  res = \
    da.blockwise(
        h_0,
        '',
        y, 'i',
        y, 'i',
        dtype=np.float32)
  res = res.compute()
  return res

homogeneity_score_dask = partial(dask_map_zip, metrics.homogeneity_score)
completeness_score_dask = partial(dask_map_zip, metrics.completeness_score)
adjusted_rand_score_dask = partial(dask_map_zip, metrics.adjusted_rand_score)
##
