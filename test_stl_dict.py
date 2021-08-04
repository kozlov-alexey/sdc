import sdc
from numba import njit, types, prange
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
from sdc.extensions.sdc_stl_hashmap_ext import _hashmap_dump
@njit
def test_impl_3():
    idx1 = pd.Int64Index([1, 2, 3, 1, 2, 3])
    a_res = sdc_indexes_map_positions(idx1)
    return len(a_res)

@njit
def test_impl_4():
    idx1 = pd.Int64Index([1, 2, 3, 1, 2, 3])
    a_res = sdc_indexes_map_positions(idx1)
    res = a_res.get(3, -1)
    return res

@njit(parallel=True)
def test_impl_5(index_data, searched):

    a_index = pd.Int64Index(index_data)
    map_pos = sdc_indexes_map_positions(a_index)
    size = len(searched)
    indexer = np.empty(size, dtype=np.int64)
    for i in prange(size):
        indexer[i] = map_pos.get(searched[i], -1)

    return indexer

def get_data(n):
    index_values = np.arange(n, dtype='int64')
    reindex_by = np.copy(index_values)
    np.random.shuffle(reindex_by)
    return [index_values, reindex_by]

print(test_impl_5(*get_data(1000)))
