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

#include <cstdint>
#include <utility>
#include <memory>
#include <type_traits>
#include <unordered_map>

#ifdef SDC_DEBUG_NATIVE
#include <iostream>
#endif

#include "utils.hpp"
#include "numba/core/runtime/nrt_external.h"


using voidptr_hash_type = size_t (*)(void* key_ptr);
using voidptr_eq_type = bool (*)(void* lhs_ptr, void* rhs_ptr);
using voidptr_refcnt = void (*)(void* key_ptr);

using iter_state = std::pair<void*, void*>;


struct Int64TrivialHash {
public:
    Int64TrivialHash() = default;
    Int64TrivialHash(const Int64TrivialHash&) = default;
    ~Int64TrivialHash() = default;
    size_t operator()(const int64_t& val) const {
        return (size_t)val;
    }
};


class CustomVoidPtrHasher
{
private:
    voidptr_hash_type ptr_hash_callback;

public:
    CustomVoidPtrHasher(void* ptr_func) {
        ptr_hash_callback = reinterpret_cast<voidptr_hash_type>(ptr_func);
    }

    size_t operator()(void* data_ptr) const {
        auto res = ptr_hash_callback(data_ptr);
        return res;
    }
};


class CustomVoidPtrEquality
{
private:
    voidptr_eq_type ptr_eq_callback;

public:
    CustomVoidPtrEquality(void* ptr_func) {
        ptr_eq_callback = reinterpret_cast<voidptr_eq_type>(ptr_func);
    }

    size_t operator()(void* lhs, void* rhs) const {
        return ptr_eq_callback(lhs, rhs);
    }
};


struct VoidPtrTypeInfo {
    voidptr_refcnt incref;
    voidptr_refcnt decref;
    uint64_t size;

    VoidPtrTypeInfo(void* incref_addr, void* decref_addr, uint64_t val_size) {
        incref = reinterpret_cast<voidptr_refcnt>(incref_addr);
        decref = reinterpret_cast<voidptr_refcnt>(decref_addr);
        size = val_size;
    }

    VoidPtrTypeInfo() = delete;
    VoidPtrTypeInfo(const VoidPtrTypeInfo&) = default;
    VoidPtrTypeInfo& operator=(const VoidPtrTypeInfo&) = default;
    VoidPtrTypeInfo& operator=(VoidPtrTypeInfo&&) = default;
    ~VoidPtrTypeInfo() = default;

    void delete_voidptr(void* ptr_data) {
        this->decref(ptr_data);
        free(ptr_data);
    }
};


template<typename Key,
         typename Val,
         typename Hash=std::hash<Key>,
         typename Equality=std::equal_to<Key>
>
class GenericHashmapBase {
public:
    using map_type = typename std::unordered_map<Key, Val, Hash, Equality>;
    using iterator_type = typename map_type::iterator;
    map_type map;

    // FIXME: 0 default size is suboptimal, can we optimize this?
    GenericHashmapBase() : map(0, Hash(), Equality()) {}
    GenericHashmapBase(const Hash& hash, const Equality& equal) : map(0, hash, equal) {}

    GenericHashmapBase(const GenericHashmapBase&) = delete;
    GenericHashmapBase& operator=(const GenericHashmapBase&) = delete;
    GenericHashmapBase(GenericHashmapBase&& rhs) = delete;
    GenericHashmapBase& operator=(GenericHashmapBase&& rhs) = delete;
    virtual ~GenericHashmapBase() {
    }

    uint64_t size() {
        return this->map.size();
    }

    int8_t contains(Key key) {
        auto it = this->map.find(key);
        return it != this->map.end();
    }

    int8_t lookup(Key key, Val* res) {
        auto result = this->map.find(key);
        bool found = result != this->map.end();
        if (found)
            *res = result->second;

        return found;
    }

    virtual void set(Key key, Val val) = 0;

    void update(GenericHashmapBase& other) {
        for (auto&& i : other.map) {
            this->set(i.first, i.second);
        }
    }

    void* getiter() {
        auto p_it = new iterator_type(this->map.begin());
        auto state = new iter_state((void*)p_it, (void*)this);
        return state;
    }
};


/* primary template for GenericHashmapType */
template<typename Key,
         typename Val,
         typename Hash=std::hash<Key>,
         typename Equality=std::equal_to<Key>
>
class GenericHashmapType : public GenericHashmapBase<Key, Val, Hash, Equality> {
public:
    // TO-DO: make VoidPtrTypeInfo templates and unify modifiers impl via calls to template funcs
    using map_type = typename GenericHashmapBase<Key, Val, Hash, Equality>::map_type;
    VoidPtrTypeInfo key_info;
    VoidPtrTypeInfo val_info;

    GenericHashmapType(const VoidPtrTypeInfo& ki,
                       const VoidPtrTypeInfo& vi,
                       const Hash& hash,
                       const Equality& equality)
    : GenericHashmapBase<Key, Val, Hash, Equality>(hash, equality),
      key_info(ki),
      val_info(vi) {}
    GenericHashmapType(const VoidPtrTypeInfo& ki, const VoidPtrTypeInfo& vi) : GenericHashmapType(ki, vi, Hash(), Equality()) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs) = delete;
    GenericHashmapType& operator=(GenericHashmapType&& rhs) = delete;
    virtual ~GenericHashmapType() {};


    void clear() {
        this->map.clear();
    }

    int8_t pop(Key key, void* res) {
        bool found = false;
        {
            auto result = this->map.find(key);
            found = result != this->map.end();
            if (found)
            {
                memcpy(res, &(result->second), this->val_info.size);
                this->map.erase(result);
            }
        }

        return found;
    }

    void set(Key key, Val val) {
        typename map_type::value_type inserted_node(key, val);
        {
            // FIXME: need to use insert_or_assign for deterministic result?
            this->map.insert(inserted_node);
        }
    }
};


/* generic-value partial specialization */
template<typename Key, typename Hash, typename Equality>
class GenericHashmapType<Key, void*, Hash, Equality> : public GenericHashmapBase<Key, void*, Hash, Equality> {
public:
    using map_type = typename GenericHashmapBase<Key, void*, Hash, Equality>::map_type;
    VoidPtrTypeInfo key_info;
    VoidPtrTypeInfo val_info;

    GenericHashmapType(const VoidPtrTypeInfo& ki,
                       const VoidPtrTypeInfo& vi,
                       const Hash& hash,
                       const Equality& equal)
    : GenericHashmapBase<Key, void*, Hash,Equality >(hash, equal),
      key_info(ki),
      val_info(vi) {}
    GenericHashmapType(const VoidPtrTypeInfo& ki, const VoidPtrTypeInfo& vi) : GenericHashmapType(ki, vi, Hash(), Equality()) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs) = delete;
    GenericHashmapType& operator=(GenericHashmapType&& rhs) = delete;
    virtual ~GenericHashmapType() {};

    void clear();
    virtual void set(Key key, void* val) override;
    int8_t pop(Key key, void* val);
};


/* generic-key partial specialization */
template<typename Val>
class GenericHashmapType<void*, Val, CustomVoidPtrHasher, CustomVoidPtrEquality> : public GenericHashmapBase<void*, Val, CustomVoidPtrHasher, CustomVoidPtrEquality> {
public:
    using map_type = typename GenericHashmapBase<void*, Val, CustomVoidPtrHasher, CustomVoidPtrEquality>::map_type;
    VoidPtrTypeInfo key_info;
    VoidPtrTypeInfo val_info;

    GenericHashmapType(const VoidPtrTypeInfo& ki,
                       const VoidPtrTypeInfo& vi,
                       const CustomVoidPtrHasher& hash,
                       const CustomVoidPtrEquality& equal)
    : GenericHashmapBase<void*, Val, CustomVoidPtrHasher, CustomVoidPtrEquality>(hash, equal),
      key_info(ki),
      val_info(vi) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs) = delete;
    GenericHashmapType& operator=(GenericHashmapType&& rhs) = delete;
    virtual ~GenericHashmapType() {};

    void clear();
    virtual void set(void* key, Val val) override;
    int8_t pop(void* key, void* val);
};


/* generic-key-and-value partial specialization */
template<>
class GenericHashmapType<void*, void*, CustomVoidPtrHasher, CustomVoidPtrEquality> : public GenericHashmapBase<void*, void*, CustomVoidPtrHasher, CustomVoidPtrEquality> {
public:
    using map_type = typename GenericHashmapBase<void*, void*, CustomVoidPtrHasher, CustomVoidPtrEquality>::map_type;
    VoidPtrTypeInfo key_info;
    VoidPtrTypeInfo val_info;

    GenericHashmapType(const VoidPtrTypeInfo& ki,
                       const VoidPtrTypeInfo& vi,
                       const CustomVoidPtrHasher& hash,
                       const CustomVoidPtrEquality& equal)
    : GenericHashmapBase<void*, void*, CustomVoidPtrHasher, CustomVoidPtrEquality>(hash, equal),
      key_info(ki),
      val_info(vi) {}

    GenericHashmapType() = delete;
    GenericHashmapType(const GenericHashmapType&) = delete;
    GenericHashmapType& operator=(const GenericHashmapType&) = delete;
    GenericHashmapType(GenericHashmapType&& rhs) = delete;
    GenericHashmapType& operator=(GenericHashmapType&& rhs) = delete;
    virtual ~GenericHashmapType() {};

    void clear();
    virtual void set(void* key, void* val) override;
    int8_t pop(void* key, void* val);
};


template <typename Key, typename Val, typename Hash=std::hash<Key>, typename Equality=std::equal_to<Key>>
using numeric_hashmap = GenericHashmapType<Key, Val, Hash, Equality>;

template <typename Val>
using generic_key_hashmap = GenericHashmapType<void*, Val, CustomVoidPtrHasher, CustomVoidPtrEquality>;

template <typename Key, typename Hash=std::hash<Key>, typename Equality=std::equal_to<Key>>
using generic_value_hashmap = GenericHashmapType<Key, void*, Hash, Equality>;

using generic_hashmap = GenericHashmapType<void*, void*, CustomVoidPtrHasher, CustomVoidPtrEquality>;


// FIXME: hasher should not be hardcoded of course! need some other way to remember what hasher/equality
// was used, as this information cannot be derived from key_type and value_types!
template<typename key_type, typename val_type>
numeric_hashmap<key_type, val_type, Int64TrivialHash>*
reinterpet_hashmap_ptr(void* p_hash_map,
                       typename std::enable_if<
                           !std::is_same<key_type, void*>::value &&
                           !std::is_same<val_type, void*>::value>::type* = 0)
{
    return reinterpret_cast<numeric_hashmap<key_type, val_type, Int64TrivialHash>*>(p_hash_map);
}

template<typename key_type, typename val_type>
generic_hashmap*
reinterpet_hashmap_ptr(void* p_hash_map,
                       typename std::enable_if<
                           std::is_same<key_type, void*>::value &&
                           std::is_same<val_type, void*>::value>::type* = 0)
{
    return reinterpret_cast<generic_hashmap*>(p_hash_map);
}

template<typename key_type, typename val_type>
generic_value_hashmap<key_type>*
reinterpet_hashmap_ptr(void* p_hash_map,
                       typename std::enable_if<
                           !std::is_same<key_type, void*>::value &&
                           std::is_same<val_type, void*>::value>::type* = 0)
{
    return reinterpret_cast<generic_value_hashmap<key_type>*>(p_hash_map);
}

template<typename key_type, typename val_type>
generic_key_hashmap<val_type>*
reinterpet_hashmap_ptr(void* p_hash_map,
                       typename std::enable_if<
                           std::is_same<key_type, void*>::value &&
                           !std::is_same<val_type, void*>::value>::type* = 0)
{
    return reinterpret_cast<generic_key_hashmap<val_type>*>(p_hash_map);
}


template <typename value_type>
void delete_generic_key_hashmap(void* p_hash_map)
{
    auto p_hash_map_spec = (generic_key_hashmap<value_type>*)p_hash_map;
    for (auto kv_pair: p_hash_map_spec->map) {
        p_hash_map_spec->key_info.delete_voidptr(kv_pair.first);
    }
    delete p_hash_map_spec;
}

template <typename key_type>
void delete_generic_value_hashmap(void* p_hash_map)
{

    auto p_hash_map_spec = (generic_value_hashmap<key_type>*)p_hash_map;
    for (auto kv_pair: p_hash_map_spec->map) {
        p_hash_map_spec->val_info.delete_voidptr(kv_pair.second);
    }
    delete p_hash_map_spec;
}

void delete_generic_hashmap(void* p_hash_map)
{
    auto p_hash_map_spec = (generic_hashmap*)p_hash_map;
    for (auto kv_pair: p_hash_map_spec->map) {
        p_hash_map_spec->key_info.delete_voidptr(kv_pair.first);
        p_hash_map_spec->val_info.delete_voidptr(kv_pair.second);
    }
    delete p_hash_map_spec;
}

template <typename key_type, typename value_type, typename Hash=std::hash<key_type>, typename Equality=std::equal_to<key_type>>
void delete_numeric_hashmap(void* p_hash_map)
{

    auto p_hash_map_spec = (numeric_hashmap<key_type, value_type, Hash, Equality>*)p_hash_map;
    delete p_hash_map_spec;
}


template <typename key_type, typename value_type>
void delete_iter_state(void* p_iter_state)
{
    auto p_iter_state_spec = reinterpret_cast<iter_state*>(p_iter_state);
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, value_type>(p_iter_state_spec->second);
    using itertype = typename std::remove_reference<decltype(*p_hash_map_spec)>::type::iterator_type;
    auto p_hash_map_iter = reinterpret_cast<itertype*>(p_iter_state_spec->first);

    delete p_hash_map_iter;
    delete p_iter_state_spec;
}


template<typename Key,
         typename Hash,
         typename Equality
>
void GenericHashmapType<Key, void*, Hash, Equality>::set(Key key, void* val)
{
    auto vsize = this->val_info.size;
    void* _val = malloc(vsize);
    memcpy(_val, val, vsize);

    typename map_type::value_type inserted_node(key, _val);
    auto res_pair = this->map.insert(inserted_node);
    if (res_pair.second)
    {
        // insertion succeeded need to incref value
        this->val_info.incref(val);
    }
    else
    {
        // insertion failed key already exists
        auto existing_node = res_pair.first;
        this->val_info.delete_voidptr(existing_node->second);
        existing_node->second = _val;
        this->val_info.incref(val);
    }
}

template<typename Key,
         typename Hash,
         typename Equality
>
void GenericHashmapType<Key, void*, Hash, Equality>::clear()
{
    for (auto kv_pair: this->map) {
        this->val_info.delete_voidptr(kv_pair.second);
    }
    this->map.clear();
}


template<typename Key,
         typename Hash,
         typename Equality
>
int8_t GenericHashmapType<Key, void*, Hash, Equality>::pop(Key key, void* res) {
    auto result = this->map.find(key);
    bool found = result != this->map.end();
    if (found)
    {
        memcpy(res, result->second, this->val_info.size);
        free(result->second);
        // no decref for value since it would be returned (and no incref on python side!)
        this->map.erase(result);
    }

    return found;
}


template<typename Val>
void GenericHashmapType<void*, Val, CustomVoidPtrHasher, CustomVoidPtrEquality>::set(void* key, Val val)
{
    auto ksize = this->key_info.size;
    void* _key = malloc(ksize);
    memcpy(_key, key, ksize);

    typename map_type::value_type inserted_node(_key, val);
    auto res_pair = this->map.insert(inserted_node);
    if (res_pair.second)
    {
        // insertion succeeded need to incref key
        this->key_info.incref(key);
    }
    else
    {
        // insertion failed key already exists
        auto existing_node = res_pair.first;
        free(_key);
        existing_node->second = val;
    }
}

template<typename Val>
void GenericHashmapType<void*, Val, CustomVoidPtrHasher, CustomVoidPtrEquality>::clear()
{
    for (auto kv_pair: this->map) {
        this->key_info.delete_voidptr(kv_pair.first);
    }
    this->map.clear();
}

template<typename Val>
int8_t GenericHashmapType<void*, Val, CustomVoidPtrHasher, CustomVoidPtrEquality>::pop(void* key, void* res) {
    bool found = false;
    {
        auto result = this->map.find(key);
        bool found = result != this->map.end();
        if (found)
        {
            memcpy(res, &(result->second), this->val_info.size);
            this->key_info.delete_voidptr(result->first);
            // no decref for value since it would be returned (and no incref on python side!)
            this->map.erase(result);
        }
    }

    return found;
}


void GenericHashmapType<void*, void*, CustomVoidPtrHasher, CustomVoidPtrEquality>::set(void* key, void* val)

{
    auto ksize = this->key_info.size;
    void* _key = malloc(ksize);
    memcpy(_key, key, ksize);

    auto vsize = this->val_info.size;
    void* _val = malloc(vsize);
    memcpy(_val, val, vsize);

    // typename map_type::value_type inserted_node(_key, _val);
    std::pair<void*, void*> inserted_node(_key, _val);
    auto res_pair = this->map.insert(inserted_node);
    if (res_pair.second)
    {
        this->key_info.incref(key);
        this->val_info.incref(val);
    }
    else
    {
        // insertion failed key already exists
        auto existing_node = res_pair.first;
        free(_key);

        this->val_info.delete_voidptr(existing_node->second);
        existing_node->second = _val;
        this->val_info.incref(val);
    }
}

void GenericHashmapType<void*, void*, CustomVoidPtrHasher, CustomVoidPtrEquality>::clear()
{
    for (auto kv_pair: this->map) {
        this->key_info.delete_voidptr(kv_pair.first);
        this->val_info.delete_voidptr(kv_pair.second);
    }
    this->map.clear();
}


int8_t GenericHashmapType<void*, void*, CustomVoidPtrHasher, CustomVoidPtrEquality>::pop(void* key, void* res) {
    bool found = false;
    {
        auto result = this->map.find(key);
        found = result != this->map.end();
        if (found)
        {
            memcpy(res, result->second, this->val_info.size);

            free(result->second);
            this->key_info.delete_voidptr(result->first);
            // no decref for value since it would be returned (and no incref on python side!)
            this->map.erase(result);
        }
    }

    return found;
}


template<typename key_type, typename val_type>
void hashmap_create(NRT_MemInfo** meminfo,
                    void* nrt_table,
                    int8_t gen_key,
                    int8_t gen_val,
                    void* hash_func_ptr,
                    void* eq_func_ptr,
                    void* key_incref_func_ptr,
                    void* key_decref_func_ptr,
                    void* val_incref_func_ptr,
                    void* val_decref_func_ptr,
                    uint64_t key_size,
                    uint64_t val_size)
{
    auto nrt = (NRT_api_functions*)nrt_table;

    // it is essential for all specializations to have common ctor signature, taking both key_info and val_info
    // since all specializations should be instantiable with different key_type/value_type, so e.g.
    // generic_key_hashmap<val_type> with val_type = void* would match full specialization. TO-DO: consider refactoring
    auto key_info = VoidPtrTypeInfo(key_incref_func_ptr, key_decref_func_ptr, key_size);
    auto val_info = VoidPtrTypeInfo(val_incref_func_ptr, val_decref_func_ptr, val_size);
    if (gen_key && gen_val)
    {
        auto p_hash_map = new generic_hashmap(key_info, val_info, CustomVoidPtrHasher(hash_func_ptr), CustomVoidPtrEquality(eq_func_ptr));
        (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_generic_hashmap);
    }
    else if (gen_key)
    {
        auto p_hash_map = new generic_key_hashmap<val_type>(key_info, val_info, CustomVoidPtrHasher(hash_func_ptr), CustomVoidPtrEquality(eq_func_ptr));
        (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_generic_key_hashmap<val_type>);
    }
    else if (gen_val)
    {
        auto p_hash_map = new generic_value_hashmap<key_type>(key_info, val_info);
        (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_generic_value_hashmap<key_type>);
    }
    else
    {
        auto p_hash_map = new numeric_hashmap<key_type, val_type>(key_info, val_info);
        (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_numeric_hashmap<key_type, val_type>);
    }

    return;
}


template<typename key_type, typename val_type>
uint64_t hashmap_size(void* p_hash_map)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    return p_hash_map_spec->size();
}


template<typename key_type, typename val_type>
void hashmap_set(void* p_hash_map, key_type key, val_type val)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    p_hash_map_spec->set(key, val);
}


template<typename key_type, typename val_type>
int8_t hashmap_lookup(void* p_hash_map,
                      key_type key,
                      val_type* res)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    return p_hash_map_spec->lookup(key, res);
}


template<typename key_type, typename val_type>
void hashmap_clear(void* p_hash_map)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    p_hash_map_spec->clear();
}


template<typename key_type, typename val_type>
int8_t hashmap_unsafe_extract(void* p_hash_map, key_type key, val_type* res)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    return p_hash_map_spec->pop(key, res);
}


template<typename key_type, typename val_type>
void hashmap_numeric_from_arrays(NRT_MemInfo** meminfo, void* nrt_table, key_type* keys, val_type* values, uint64_t size)
{
    auto nrt = (NRT_api_functions*)nrt_table;
    auto key_info = VoidPtrTypeInfo(nullptr, nullptr, sizeof(key_type));
    auto val_info = VoidPtrTypeInfo(nullptr, nullptr, sizeof(val_type));
    auto p_hash_map = new numeric_hashmap<key_type, val_type, Int64TrivialHash>(key_info, val_info);
    (*meminfo) = nrt->manage_memory((void*)p_hash_map, delete_numeric_hashmap<key_type, val_type>);

     for(size_t i=0; i!=size; ++i) {
         auto kv_pair = std::pair<const key_type, val_type>(keys[i], values[i]);
         p_hash_map->map.insert(
             std::move(kv_pair)
         );
     }
}


template<typename key_type, typename val_type>
void hashmap_update(void* p_self_hash_map, void* p_arg_hash_map)
{
    auto p_self_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_self_hash_map);
    auto p_arg_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_arg_hash_map);
    p_self_hash_map_spec->update(*p_arg_hash_map_spec);
    return;
}


#ifdef SDC_DEBUG_NATIVE
template<typename key_type, typename val_type>
void hashmap_dump(void* p_hash_map)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    auto size = p_hash_map_spec->map.size();
    std::cout << "Hashmap at: " << p_hash_map_spec << ", size = " << size << std::endl;
    for (auto kv_pair: p_hash_map_spec->map)
    {
        std::cout << "key, value: " << kv_pair.first << ", " << kv_pair.second << std::endl;
    }
    return;
}
#endif


template<typename key_type, typename val_type>
void* hashmap_getiter(NRT_MemInfo** meminfo, void* nrt_table, void* p_hash_map)
{
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_hash_map);
    auto p_iter_state = p_hash_map_spec->getiter();

    auto nrt = (NRT_api_functions*)nrt_table;
    (*meminfo) = nrt->manage_memory((void*)p_iter_state, delete_iter_state<key_type, val_type>);
    return p_iter_state;
}


template<typename key_type, typename val_type>
int8_t hashmap_iternext(void* p_iter_state, key_type* ret_key, val_type* ret_val)
{
    auto p_iter_state_spec = reinterpret_cast<iter_state*>(p_iter_state);
    auto p_hash_map_spec = reinterpet_hashmap_ptr<key_type, val_type>(p_iter_state_spec->second);
    using itertype = typename std::remove_reference<decltype(*p_hash_map_spec)>::type::iterator_type;
    auto p_hash_map_iter = reinterpret_cast<itertype*>(p_iter_state_spec->first);

    int8_t status = 1;
    if (*p_hash_map_iter != p_hash_map_spec->map.end())
    {
        *ret_key = (*p_hash_map_iter)->first;
        *ret_val = (*p_hash_map_iter)->second;
        status = 0;
        ++(*p_hash_map_iter);
    }

    return status;
}
