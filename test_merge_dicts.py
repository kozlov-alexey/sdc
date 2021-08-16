import sdc
from numba import njit, types, prange
import numpy as np
from sdc.extensions.sdc_stl_hashmap_ext import SdcDict
from sdc.extensions.sdc_hashmap_ext import ConcurrentDict
from numba.typed import Dict, List

import time
from sdc.utilities.prange_utils import parallel_chunks, get_pool_size, get_chunks
from numba.misc.special import literal_unroll

dict_type = types.DictType(types.int64, types.int64)

''' This version shows that building parts of dicts can be done in parallel with
prange and would take 0.33s on 8 threads! '''
@njit(parallel=True)
def test_impl_1(index_data, searched):

    t1 = time.time()
    n_chunks = get_pool_size()
#     dict_parts = List.empty_list(dict_type)
#     for i in range(n_chunks):
#         dict_parts.append(
#             Dict.empty(types.int64, types.int64)
#         )

    dict_parts = (
        Dict.empty(types.int64, types.int64),
        Dict.empty(types.int64, types.int64),
        Dict.empty(types.int64, types.int64),
        Dict.empty(types.int64, types.int64),
        Dict.empty(types.int64, types.int64),
        Dict.empty(types.int64, types.int64),
        Dict.empty(types.int64, types.int64),
        Dict.empty(types.int64, types.int64)
    )
    prange_chunks = parallel_chunks(len(index_data))
#     print("DEBUG: ", n_chunks, prange_chunks)
    for i in prange(n_chunks):
        chunk = prange_chunks[i]
#         print("DEBUG: ", i, chunk.start, chunk.stop, dict_part)
        for j in range(chunk.start, chunk.stop):
            dict_parts[i][index_data[j]] = j
#         print("DEBUG: after ", i, dict_part)

    # print("DEBUG: ", dict_parts)
#     res_dict = ConcurrentDict.empty(types.int64, types.int64)
#     for i in prange(n_chunks):
#         for k, v in dict_parts[i].items():
#             res_dict[k] = v
    t2 = time.time()
    return len(dict_parts[0]), t2 - t1

@njit(parallel=False, inline='never')
def find_index(dict_parts, key):
    res = -1
    size = len(dict_parts)
#     size = 6    ## there's no difference between size=6 and size=7 why??
#     size = 7

    for j in range(size):
        if res == -1:
            res = dict_parts[j].get(key, -1)

#     found_val = dict_parts[0].get(key, -1)
#     res = res if found_val == -1 else found_val
#     found_val = dict_parts[1].get(key, -1)
#     res = res if found_val == -1 else found_val
#     found_val = dict_parts[2].get(key, -1)
#     res = res if found_val == -1 else found_val
    return res


''' But if we use ConcurrentDict it doesn't scale and subsequent prange doesn't get parallelized
(again that old problem with ConcurrentDict ctor breaking ParAcc) '''
@njit(parallel=True)
def test_impl_2(index_data, searched):

    # n_chunks = get_pool_size()
    n_chunks = 2

    t1 = time.time()
    dict_parts = [Dict.empty(types.int64, types.int64) for _ in range(n_chunks)]

#     dict_parts = List.empty_list(dict_type)
#     for i in range(n_chunks):
#         dict_parts.append(
#             Dict.empty(types.int64, types.int64)
#         )

#     dict_parts = (
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64)
#     )
    prange_chunks = get_chunks(len(index_data), n_chunks)
    for i in prange(n_chunks):
        chunk = prange_chunks[i]
        dict_part = dict_parts[i]
        for j in range(chunk.start, chunk.stop):
            dict_part[index_data[j]] = j

    indexer = np.zeros(len(searched), dtype=np.int64) - 1
    for i in prange(len(searched)):
        target_val = searched[i]
        indexer[i] = find_index(dict_parts, target_val)
#         for j in range(n_chunks):
#             if indexer[i] == -1:
#                 indexer[i] = dict_parts[j].get(target_val, -1)

    t2 = time.time()
    return len(indexer), t2 - t1


@njit(parallel=True)
def test_impl_3(index_data, searched):

#     n_chunks = get_pool_size()
    n_chunks = 2
#     n_chunks = max(get_pool_size() // 10, 2)

    t1 = time.time()
    dict_parts = [Dict.empty(types.int64, types.int64) for _ in range(n_chunks)]
    dict_parts_info = np.empty((n_chunks, 2), dtype=np.int64)

#     dict_parts = List.empty_list(dict_type)
#     for i in range(n_chunks):
#         dict_parts.append(
#             Dict.empty(types.int64, types.int64)
#         )

#     dict_parts = (
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64)
#     )
    prange_chunks = get_chunks(len(index_data), n_chunks)
    for i in prange(n_chunks):
        chunk = prange_chunks[i]
        dict_part = dict_parts[i]
        dict_parts_info[i][0] = index_data[chunk.start]
        dict_parts_info[i][1] = index_data[chunk.start]
        for j in range(chunk.start, chunk.stop):
            new_key = index_data[j]
            dict_part[new_key] = j
            if new_key < dict_parts_info[i][0]:
                dict_parts_info[i][0] = new_key
            if new_key > dict_parts_info[i][1]:
                dict_parts_info[i][1] = new_key

    indexer = np.empty(len(searched), dtype=np.int64)
#     all_indexers = np.empty((len(searched), n_chunks), dtype=np.int64)
#     for i in prange(len(searched)):
#         target_val = searched[i]
#         for j in range(n_chunks):
#             all_indexers[i][j] = dict_parts[j].get(target_val, -1)
#         indexer[i] = np.max(all_indexers[i])

    for i in prange(len(searched)):
        target_val = searched[i]
        indexer[i] = -1
        for j in range(n_chunks):
            if (dict_parts_info[j][0] <= target_val and
                    target_val <= dict_parts_info[j][1] and
                    indexer[i] == -1):
                indexer[i] = dict_parts[j].get(target_val, -1)

    '''
    merged_dict = dict_parts[0]
    for i in range(1, n_chunks):
        merged_dict.update(dict_parts[i])
    for i in prange(len(searched)):
        target_val = searched[i]
        indexer[i] = merged_dict.get(target_val, -1)
    '''

    t2 = time.time()

    return indexer, t2 - t1


'''
@njit(parallel=True)
def test_impl_4(index_data, searched):

#     n_chunks = get_pool_size()
#     n_chunks = 2
    n_chunks = 1
#     n_chunks = max(get_pool_size() // 10, 2)

    t1 = time.time()
    dict_parts = [Dict.empty(types.int64, types.int64) for _ in range(n_chunks)]
    dict_parts_info = np.empty((n_chunks, 2), dtype=np.int64)

#     dict_parts = List.empty_list(dict_type)
#     for i in range(n_chunks):
#         dict_parts.append(
#             Dict.empty(types.int64, types.int64)
#         )

#     dict_parts = (
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64),
#         Dict.empty(types.int64, types.int64)
#     )
    prange_chunks = get_chunks(len(index_data), n_chunks)
    for i in prange(n_chunks):
        chunk = prange_chunks[i]
        dict_part = dict_parts[i]
        dict_parts_info[i][0] = index_data[chunk.start]
        dict_parts_info[i][1] = index_data[chunk.start]
        for j in range(chunk.start, chunk.stop):
            new_key = index_data[j]
            dict_part[new_key] = j
            if new_key < dict_parts_info[i][0]:
                dict_parts_info[i][0] = new_key
            if new_key > dict_parts_info[i][1]:
                dict_parts_info[i][1] = new_key

    indexer = np.empty(len(searched), dtype=np.int64)
#     all_indexers = np.empty((len(searched), n_chunks), dtype=np.int64)
#     for i in prange(len(searched)):
#         target_val = searched[i]
#         for j in range(n_chunks):
#             all_indexers[i][j] = dict_parts[j].get(target_val, -1)
#         indexer[i] = np.max(all_indexers[i])

    for i in prange(len(searched)):
        target_val = searched[i]
        indexer[i] = -1
        for j in range(n_chunks):
            if (dict_parts_info[j][0] <= target_val and
                    target_val <= dict_parts_info[j][1] and
                    indexer[i] == -1):
                indexer[i] = dict_parts[j].get(target_val, -1)

    t2 = time.time()

    return indexer, t2 - t1
'''


@njit(parallel=True)
def test_impl_4(index_data, searched):

    t1 = time.time()
    map_positions = Dict.empty(types.int64, types.int64)
    for i, label in enumerate(index_data):
        map_positions[label] = i

#     map_positions = ConcurrentDict.empty(types.int64, types.int64)
#     for i in prange(len(index_data)):
#         map_positions[index_data[i]] = i

#     map_positions = ConcurrentDict.from_arrays(index_data, np.arange(len(index_data)))

    indexer = np.empty(len(searched), dtype=np.int64)

    for i in prange(len(searched)):
        target_val = searched[i]
        indexer[i] = map_positions.get(target_val, -1)

    t2 = time.time()

    return indexer, t2 - t1


from sdc.extensions.sdc_stl_hashmap_ext import fill_indexer_cpp_stl
@njit(parallel=True)
def test_stl_indexer(index_data, searched):
    a_index = pd.Int64Index(index_data)
    t1 = time.time()
    indexer = np.empty(len(searched), dtype=np.int64)
    fill_indexer_cpp_stl(index_data, searched, indexer)
    t2 = time.time()
    return indexer, t2 - t1




def get_data(n):
    index_values = np.arange(n, dtype='int64')
    reindex_by = np.copy(index_values)
    np.random.shuffle(index_values)
    np.random.shuffle(reindex_by)
    return [index_values, reindex_by]

import pandas as pd
def reference_impl(index_data, searched):
    t1 = time.time()
    a_index = pd.Int64Index(index_data)
    indexer = a_index.get_indexer(searched)
    t2 = time.time()
    return indexer, t2 - t1

# n = 10
n = 10000000
args_data = get_data(n)

reference_res = reference_impl(*args_data)
# print(reference_res[1], reference_res[0])
print(reference_res[1])

# print("args_data:", args_data)
res = test_stl_indexer(*args_data)
print(res[1])
# print(res[1], res[0])

assert np.array_equal(res[0], reference_res[0])
