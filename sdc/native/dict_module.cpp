// *****************************************************************************
// Copyright (c) 2021, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//     Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// *****************************************************************************

#include <Python.h>
#include "hashmap.hpp"
#include <chrono>
#include <mutex>

class TrivialTBBHashCompare {
public:
    static size_t hash(const int64_t& val) {
        return (size_t)val;
    }
    static bool equal(const int64_t& k1, const int64_t& k2) {
        return k1==k2;
    }
};

using namespace std::chrono;

#define declare_hashmap_create(key_type, val_type, suffix) \
void hashmap_create_##suffix(NRT_MemInfo** meminfo, \
                             void* nrt_table, \
                             int8_t gen_key, \
                             int8_t gen_val, \
                             void* hash_func_ptr, \
                             void* eq_func_ptr, \
                             void* key_incref_func_ptr, \
                             void* key_decref_func_ptr, \
                             void* val_incref_func_ptr, \
                             void* val_decref_func_ptr, \
                             uint64_t key_size, \
                             uint64_t val_size) \
{ \
    hashmap_create<key_type, val_type>( \
                meminfo, nrt_table, \
                gen_key, gen_val, \
                hash_func_ptr, eq_func_ptr, \
                key_incref_func_ptr, key_decref_func_ptr, \
                val_incref_func_ptr, val_decref_func_ptr, \
                key_size, val_size); \
} \


#define declare_hashmap_size(key_type, val_type, suffix) \
uint64_t hashmap_size_##suffix(void* p_hash_map) \
{ \
    return hashmap_size<key_type, val_type>(p_hash_map); \
} \


#define declare_hashmap_set(key_type, val_type, suffix) \
void hashmap_set_##suffix(void* p_hash_map, key_type key, val_type val) \
{ \
    hashmap_set<key_type, val_type>(p_hash_map, key, val); \
} \


#define declare_hashmap_contains(key_type, val_type, suffix) \
int8_t hashmap_contains_##suffix(void* p_hash_map, key_type key) \
{ \
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map); \
    return p_hash_map_spec->contains(key); \
} \


#define declare_hashmap_lookup(key_type, val_type, suffix) \
int8_t hashmap_lookup_##suffix(void* p_hash_map, key_type key, val_type* res) \
{ \
    return hashmap_lookup<key_type, val_type>(p_hash_map, key, res); \
} \


#define declare_hashmap_clear(key_type, val_type, suffix) \
void hashmap_clear_##suffix(void* p_hash_map) \
{ \
    return hashmap_clear<key_type, val_type>(p_hash_map); \
} \


#define declare_hashmap_pop(key_type, val_type, suffix) \
int8_t hashmap_pop_##suffix(void* p_hash_map, key_type key, val_type* res) \
{ \
    return hashmap_unsafe_extract<key_type, val_type>(p_hash_map, key, res); \
} \


#define declare_hashmap_create_from_data(key_type, val_type) \
void hashmap_create_from_data_##key_type##_to_##val_type(NRT_MemInfo** meminfo, void* nrt_table, key_type* keys, val_type* values, int64_t size) \
{ \
    hashmap_numeric_from_arrays<key_type, val_type>(meminfo, nrt_table, keys, values, size); \
} \


#define declare_hashmap_update(key_type, val_type, suffix) \
void hashmap_update_##suffix(void* p_self_hash_map, void* p_arg_hash_map) \
{ \
    return hashmap_update<key_type, val_type>(p_self_hash_map, p_arg_hash_map); \
} \


#ifdef SDC_DEBUG_NATIVE
#define declare_hashmap_dump(key_type, val_type, suffix) \
void hashmap_dump_##suffix(void* p_hash_map) \
{ \
    hashmap_dump<key_type, val_type>(p_hash_map); \
}
#else
#define declare_hashmap_dump(key_type, val_type, suffix)
#endif


#define declare_hashmap_getiter(key_type, val_type, suffix) \
void* hashmap_getiter_##suffix(NRT_MemInfo** meminfo, void* nrt_table, void* p_hash_map) \
{ \
    return hashmap_getiter<key_type, val_type>(meminfo, nrt_table, p_hash_map); \
} \


#define declare_hashmap_iternext(key_type, val_type, suffix) \
int8_t hashmap_iternext_##suffix(void* p_iter_state, key_type* ret_key, val_type* ret_val) \
{ \
    return hashmap_iternext<key_type, val_type>(p_iter_state, ret_key, ret_val); \
} \


#define declare_hashmap_build_map_positions(key_type, suffix) \
void hashmap_build_map_positions_##suffix(NRT_MemInfo** meminfo, \
                             void* nrt_table, \
                             int8_t gen_key, \
                             int8_t gen_val, \
                             void* hash_func_ptr, \
                             void* eq_func_ptr, \
                             void* key_incref_func_ptr, \
                             void* key_decref_func_ptr, \
                             void* val_incref_func_ptr, \
                             void* val_decref_func_ptr, \
                             uint64_t key_size, \
                             uint64_t val_size) \
{ \
    hashmap_build_map_positions<key_type>( \
                meminfo, nrt_table, \
                gen_key, gen_val, \
                hash_func_ptr, eq_func_ptr, \
                key_incref_func_ptr, key_decref_func_ptr, \
                val_incref_func_ptr, val_decref_func_ptr, \
                key_size, val_size); \
} \


#define declare_hashmap(key_type, val_type, suffix) \
declare_hashmap_create(key_type, val_type, suffix); \
declare_hashmap_size(key_type, val_type, suffix); \
declare_hashmap_set(key_type, val_type, suffix); \
declare_hashmap_contains(key_type, val_type, suffix); \
declare_hashmap_lookup(key_type, val_type, suffix); \
declare_hashmap_clear(key_type, val_type, suffix); \
declare_hashmap_pop(key_type, val_type, suffix); \
declare_hashmap_update(key_type, val_type, suffix); \
declare_hashmap_getiter(key_type, val_type, suffix); \
declare_hashmap_iternext(key_type, val_type, suffix); \
declare_hashmap_dump(key_type, val_type, suffix); \


#define REGISTER(func) PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)));


#define register_release(suffix) \
REGISTER(hashmap_create_##suffix) \
REGISTER(hashmap_size_##suffix) \
REGISTER(hashmap_set_##suffix) \
REGISTER(hashmap_contains_##suffix) \
REGISTER(hashmap_lookup_##suffix) \
REGISTER(hashmap_clear_##suffix) \
REGISTER(hashmap_pop_##suffix) \
REGISTER(hashmap_update_##suffix) \
REGISTER(hashmap_getiter_##suffix) \
REGISTER(hashmap_iternext_##suffix) \

#define register_debug(suffix) \
REGISTER(hashmap_dump_##suffix)

#ifndef SDC_DEBUG_NATIVE
#define register_hashmap(suffix) register_release(suffix)
#else
#define register_hashmap(suffix) \
register_release(suffix) \
register_debug(suffix)
#endif


extern "C"
{

// declare all hashmap methods for below combinations of key-value types
declare_hashmap(int32_t, int32_t, int32_t_to_int32_t)
declare_hashmap(int32_t, int64_t, int32_t_to_int64_t)
declare_hashmap(int32_t, float, int32_t_to_float)
declare_hashmap(int32_t, double, int32_t_to_double)

declare_hashmap(int64_t, int32_t, int64_t_to_int32_t)
declare_hashmap(int64_t, int64_t, int64_t_to_int64_t)
declare_hashmap(int64_t, float, int64_t_to_float)
declare_hashmap(int64_t, double, int64_t_to_double)

declare_hashmap(void*, int32_t, voidptr_to_int32_t)
declare_hashmap(void*, int64_t, voidptr_to_int64_t)
declare_hashmap(void*, float, voidptr_to_float)
declare_hashmap(void*, double, voidptr_to_double)

declare_hashmap(int32_t, void*, int32_t_to_voidptr)
declare_hashmap(int64_t, void*, int64_t_to_voidptr)

declare_hashmap(void*, void*, voidptr_to_voidptr)

// additionally declare create_from_data functions for numeric hashmap
declare_hashmap_create_from_data(int32_t, int32_t)
declare_hashmap_create_from_data(int32_t, int64_t)
declare_hashmap_create_from_data(int32_t, float)
declare_hashmap_create_from_data(int32_t, double)

declare_hashmap_create_from_data(int64_t, int32_t)
declare_hashmap_create_from_data(int64_t, int64_t)
declare_hashmap_create_from_data(int64_t, float)
declare_hashmap_create_from_data(int64_t, double)


void hashmap_build_map_positions_int64_t(NRT_MemInfo** meminfo,
                             void* nrt_table,
                             int64_t* data,
                             uint64_t size)
{
    auto nrt = (NRT_api_functions*)nrt_table;

    // FIXME: add ctor with fixed size of buckets!
    // FIXME: do we need to use TrivialHash function (at least on MSVC we do need)
    auto key_info = VoidPtrTypeInfo(nullptr, nullptr, sizeof(int64_t));
    auto val_info = VoidPtrTypeInfo(nullptr, nullptr, sizeof(int64_t));
//    auto p_hash_map = new numeric_hashmap<int64_t, int64_t>(key_info, val_info);
//    (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_numeric_hashmap<int64_t, int64_t>);
     auto p_hash_map = new numeric_hashmap<int64_t, int64_t, Int64TrivialHash>(key_info, val_info, Int64TrivialHash(), std::equal_to<int64_t>());
     (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_numeric_hashmap<int64_t, int64_t, Int64TrivialHash>);

    for (int i=0; i < size; ++i)
        p_hash_map->set(data[i], i);
}


uint8_t hashmap_native_build_and_fill_stl(int64_t* data, int64_t* searched, int64_t size, int64_t* res)
{
    auto t1 = high_resolution_clock::now();
//    auto my_map_ptr = new std::unordered_map<int64_t, int64_t, Int64TrivialHash>(size, Int64TrivialHash());
//    auto my_map_ptr = new std::unordered_map<int64_t, int64_t>(size);
    auto my_map_ptr = new tbb::concurrent_hash_map<int64_t, int64_t, TrivialTBBHashCompare>(2*size, TrivialTBBHashCompare());
    auto& my_map = *my_map_ptr;
//    for (int i=0; i < size; ++i) {
//        my_map.emplace(data[i], i);
//    }

    utils::tbb_control::get_arena().execute([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, size),
                         [&](const tbb::blocked_range<size_t>& r) {
                             for(size_t i=r.begin(); i!=r.end(); ++i) {
                                 my_map.emplace(data[i], i);
                             }
                         }
            );
    });

    if (my_map.size() < size)
        return 0;

    auto t2 = high_resolution_clock::now();
    duration<double, std::ratio<1, 1>> ms_double = t2 - t1;
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "native (STL) building map: " << ms_int.count() << " ms, (" << ms_double.count() << " sec)" << std::endl;

//    for (int i=0; i < size; ++i) {
//        auto it = my_map.find(searched[i]);
//        if (it != my_map.end())
//            res[i] = it->second;
//        else
//            res[i] = -1;
//    }

    auto it_map_end = my_map.end();
    utils::tbb_control::get_arena().execute([&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, size),
                     [&](const tbb::blocked_range<size_t>& r) {
                         for(size_t i=r.begin(); i!=r.end(); ++i) {
//                             auto it = my_map.find(searched[i]);
//                             if (it != it_map_end)
//                                 res[i] = it->second;
//                             else
//                                 res[i] = -1;
                             auto it_pair = my_map.equal_range(searched[i]);
                             if (it_pair.first != my_map.end()) {
                                 res[i] = it_pair.first->second;
                             } else {
                                 res[i] = -1;
                             }
                         }
                     }
        );
    });

    auto t3 = high_resolution_clock::now();
    ms_double = t3 - t2;
    ms_int = duration_cast<milliseconds>(t3 - t2);
    std::cout << "native (STL) filling indexer: " << ms_int.count() << " ms, (" << ms_double.count() << " sec)" << std::endl;
    ms_double = t3 - t1;
    ms_int = duration_cast<milliseconds>(t3 - t1);
    std::cout << "total time: " << ms_int.count() << " ms, (" << ms_double.count() << " sec)" << std::endl;

    return 1;
}

void set_number_of_threads(uint64_t threads)
{
    utils::tbb_control::set_threads_num(threads);
}

PyMODINIT_FUNC PyInit_hnative_dict()
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "hnative_dict",
        "No docs",
        -1,
        NULL,
    };
    PyObject* m = PyModule_Create(&moduledef);
    if (m == NULL)
    {
        return NULL;
    }

    // register previosuly declared hashmap methods
    register_hashmap(int32_t_to_int32_t)
    register_hashmap(int32_t_to_int64_t)
    register_hashmap(int64_t_to_int32_t)
    register_hashmap(int64_t_to_int64_t)

    register_hashmap(int32_t_to_float)
    register_hashmap(int32_t_to_double)
    register_hashmap(int64_t_to_float)
    register_hashmap(int64_t_to_double)

    register_hashmap(voidptr_to_int32_t)
    register_hashmap(voidptr_to_int64_t)
    register_hashmap(voidptr_to_float)
    register_hashmap(voidptr_to_double)

    register_hashmap(int32_t_to_voidptr)
    register_hashmap(int64_t_to_voidptr)

    register_hashmap(voidptr_to_voidptr);

    // hashmap create_from_data functions for numeric hashmap
    REGISTER(hashmap_create_from_data_int32_t_to_int32_t)
    REGISTER(hashmap_create_from_data_int32_t_to_int64_t)
    REGISTER(hashmap_create_from_data_int64_t_to_int32_t)
    REGISTER(hashmap_create_from_data_int64_t_to_int64_t)

    REGISTER(hashmap_create_from_data_int32_t_to_float)
    REGISTER(hashmap_create_from_data_int32_t_to_double)
    REGISTER(hashmap_create_from_data_int64_t_to_float)
    REGISTER(hashmap_create_from_data_int64_t_to_double)

    REGISTER(hashmap_build_map_positions_int64_t)
    REGISTER(hashmap_native_build_and_fill_stl)

    REGISTER(set_number_of_threads)
    utils::tbb_control::init();

    return m;
}

}  // extern "C"

#undef declare_hashmap_create
#undef declare_hashmap_size
#undef declare_hashmap_set
#undef declare_hashmap_contains
#undef declare_hashmap_lookup
#undef declare_hashmap_clear
#undef declare_hashmap_pop
#undef declare_hashmap_create_from_data
#undef declare_hashmap_update
#undef declare_hashmap_getiter
#undef declare_hashmap_iternext
#undef declare_hashmap_dump
#undef register_hashmap
#undef REGISTER
#undef declare_hashmap
