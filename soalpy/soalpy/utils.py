import numpy as np
import pandas
pd = pandas
from time import time, sleep
import concurrent
import gc
import matplotlib.pyplot as plt

from fastai.torch_core import show_image, show_images
##
from sklearn import datasets
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.model_selection import train_test_split
#: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#: https://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html

from sklearn.preprocessing import MinMaxScaler
#: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
##
