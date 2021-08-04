import sdc
from numba import njit, types
import numpy as np
from sdc.extensions.sdc_stl_hashmap_ext import SdcDict

@njit
def test_impl_1():
    a_res = SdcDict.empty(types.int64, types.int64)
    a_res[1] = 0
    a_res[2] = 1
    a_res[1] = 1
    return len(a_res)

@njit
def test_impl_2():
    keys = np.arange(10)
    vals = np.ones(10)
    a_res = SdcDict.from_arrays(keys, vals)
    return len(a_res)

import pandas as pd
from sdc.extensions.sdc_indexing import sdc_indexes_map_positions
@njit
def test_impl_3():
    idx1 = pd.Int64Index([1, 2, 3, 1, 2, 3])
    a_res = sdc_indexes_map_positions(idx1)
    return len(a_res)


print(test_impl_3())
