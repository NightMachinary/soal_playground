import sys
import os
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
from contextlib import contextmanager

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
##
