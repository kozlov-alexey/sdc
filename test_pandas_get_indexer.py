import numba
from numba import njit
from numba.core import types
from numba import prange
from numba.typed import Dict
import sdc
import time
import numpy as np
import pandas as pd
import os

from sdc.tests.test_utils import gen_strlist

def test_get_indexer(index_data, searched):
    a_index = pd.Index(index_data)
    t1 = time.time()
    indexer = a_index.get_indexer(searched)
    t_exec = time.time() - t1
    return t_exec, indexer

def test_reindex(index_data, searched):
    a_index = pd.Index(index_data)
    t1 = time.time()
    indexer = a_index.reindex(searched)
    t_exec = time.time() - t1
    return t_exec, indexer

@njit(parallel=True)
def test_typed_dict(index_data, searched):
    t1 = time.time()
    map_pos = Dict.empty(types.int64, types.int64)
    for i, label in enumerate(index_data):
        map_pos[label] = i
    size = len(searched)
    indexer = np.empty(size, dtype=np.int64)
    for i in prange(size):
        indexer[i] = map_pos.get(searched[i], -1)
    t_exec = time.time() - t1
    return t_exec, indexer

# from sdc.extensions.sdc_indexing import sdc_indexes_map_positions
# from sdc.extensions.sdc_stl_hashmap_type import SdcDict
# @njit(parallel=True)
# def test_stl_indexer(index_data, searched):
#     a_index = pd.Int64Index(index_data)
#     t1 = time.time()
#     # map_pos = SdcDict.empty(types.int64, types.int64)
#     # for i in range(len(index_data)):
#     #     map_pos[index_data[i]] = i
# #     range_values = np.arange(len(index_data))
# #     map_pos = SdcDict.from_arrays(index_data, range_values)
#     map_pos = sdc_indexes_map_positions(a_index)
#     size = len(searched)
#     indexer = np.empty(size, dtype=np.int64)
#     for i in prange(size):
#         indexer[i] = map_pos.get(searched[i], -1)
# #         indexer[i] = map_pos[searched[i]]
#     t_exec = time.time() - t1
#     return t_exec, indexer


from sdc.extensions.sdc_indexing import sdc_indexes_map_positions
from sdc.extensions.sdc_stl_hashmap_type import SdcDict
@njit(parallel=True)
def test_stl_indexer(index_data, searched):
    a_index = pd.Int64Index(index_data)
    t1 = time.time()
    # map_pos = SdcDict.empty(types.int64, types.int64)
    # for i in range(len(index_data)):
    #     map_pos[index_data[i]] = i
#     range_values = np.arange(len(index_data))
#     map_pos = SdcDict.from_arrays(index_data, range_values)
    map_pos = sdc_indexes_map_positions(a_index)
    size = len(searched)
    indexer = np.empty(size, dtype=np.int64)
    for i in prange(size):
        indexer[i] = map_pos.get(searched[i], -1)
#         indexer[i] = map_pos[searched[i]]
    t_exec = time.time() - t1
    return t_exec, indexer


### produces data for reindexing/reordering of unique labels
def get_data(n):
    index_values = np.arange(n, dtype='int64')
    reindex_by = np.copy(index_values)
    np.random.shuffle(reindex_by)
    return [index_values, reindex_by]

### produces data for reindexing but lots of duplicate labels (smaller dict size)
# def get_data(n):
#     index_values = np.arange(n // 4, dtype='int64')
#     index_data = np.random.choice(index_values, n).astype(dtype='int64')
#     reindex_by = np.arange(n, dtype='int64')
#     np.random.shuffle(reindex_by)
#     return [index_data, reindex_by]

tested_impl_1 = test_get_indexer
tested_impl_2 = test_reindex
tested_impl_3 = test_typed_dict
tested_impl_4 = test_stl_indexer
data_gen = get_data


# all_nthreads = (1, )
all_nthreads = (8, )
# all_nthreads = (1, 2, 4, 8)                     # laptop version
# all_nthreads = (1, 2, 4, 8, 16, 28)       # Xeon version

# def data_gen(n):
# #     keys = gen_strlist(n)
#     keys = np.random.randint(-5000, 5000, n).astype('int64')
#     values = np.arange(n, dtype='int64')
#     return (keys, values)


def launcher():
    np.random.seed(0)
    all_columns = tuple([
        'tested_1',
        'tested_2',
        'tested_3',
        'tested_4',
    ])
    results = pd.DataFrame(
        {},
        columns=all_columns,
        index=pd.Index(all_nthreads, name='n_threads')
    )

    # warmup
    args = data_gen(100)
    tested_impl_1(*args)
    tested_impl_2(*args)
    tested_impl_3(*args)
    tested_impl_4(*args)

    n = 10000000    # for numeric
#     n = 500000    # for unicodes
#     n = 1000000     # this can also fit for unicodes
#     n = 100

    # args = data_gen(n)        ## using the same series for all runs is not fair
                                ## since pandas will compute and store index grouper during first run
    for n_threads in all_nthreads:
  
        args = data_gen(n)      ## this would be much fair
        numba.set_num_threads(n_threads)
        results['tested_1'][n_threads] = tested_impl_1(*args)[0]
        results['tested_2'][n_threads] = tested_impl_2(*args)[0]
        results['tested_3'][n_threads] = tested_impl_3(*args)[0]
        results['tested_4'][n_threads] = tested_impl_4(*args)[0]
  
  
    print("Results:")
    print(results)


if __name__ == '__main__':
    launcher()

## Summary: 
###    [] both numeric and string data sort_values doesn't scale at all
###       (but first is faster than pandas, second is not), why?
###    [] reindex operation is way slower than pandas too (e.g. on n = 5000000
###       tested vs reference is 1.981  0.727997 (string data in series)
###       but it's actually the same for float data as well - our reindexing is SLOW
###    [] Need to see how pandas does this reindexing, that it's so fast.



