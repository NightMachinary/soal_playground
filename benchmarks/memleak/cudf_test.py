#!/usr/bin/env python3


import numpy
np = numpy
import cudf

n = 10**(4+4)
a = np.random.uniform(size=(n), dtype=np.float32)
cudf.DataFrame(a)
