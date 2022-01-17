#!/usr/bin/env python

from brish import z, zp
import gc
import numpy
np = numpy

def some_fn():
    ## Both of these will "leak" memory.
    a = np.random.uniform(size=(n))
    ##
    # a = np.ones(n, dtype=np.int64)
    ##

    # return None
    return a

z('''
function memory-free-get {{
  local free_ram
  free_ram="$(command free --bytes | rg 'Mem: '| awkn 4)" @TRET
  ec "$free_ram" | numfmt-humanfriendly-bytes
  # command free -h
}}
''')

zp('echo -n "start: " ; memory-free-get')

n = 100_000_000
a = None
for i in range(3):
    a = some_fn()
    zp('echo -n "{i}: " ; memory-free-get')

del a
gc.collect()
zp('echo -n "end: " ; memory-free-get')

### Output
## macOS (weird):
# start: 2.5GiB
# 0: 3.4GiB
# 1: 3.5GiB
# 2: 3.5GiB
# end: 3.5GiB
## Linux:
# start: 1.7GiB
# 0: 900MiB
# 1: 911MiB
# 2: 916MiB
# end: 1.7GiB
###
