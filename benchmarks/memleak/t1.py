#!/usr/bin/env python

from brish import z, zp
import gc
import numpy
np = numpy

def some_fn():
    ## Both of these will "leak" memory.
    # a = np.random.uniform(size=(n))
    ##
    a = np.ones(n, dtype=np.int64)
    ##

    return None

zp('echo -n "start: " ; memory-free-get')

n = 200_000_000
a = None
for i in range(3):
    a = some_fn()
    zp('echo -n "{i}: " ; memory-free-get')

del a #: redundant, as we haven't returned anything.
gc.collect()
zp('echo -n "end: " ; memory-free-get')

### Output
# start: 2.5GiB
# 0: 3.4GiB
# 1: 3.5GiB
# 2: 3.5GiB
# end: 3.5GiB
###
