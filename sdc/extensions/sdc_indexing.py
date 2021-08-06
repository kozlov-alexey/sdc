# *****************************************************************************
# Copyright (c) 2019-2021, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import llvmlite.binding as ll
import llvmlite.llvmpy.core as lc
import numba
import numpy as np
import operator
import sdc

from sdc import hstr_ext
from glob import glob
from llvmlite import ir as lir
from numba import types, cfunc
from numba.core import cgutils
from numba.extending import (typeof_impl, type_callable, models, register_model, NativeValue,
                             lower_builtin, box, unbox, lower_getattr, intrinsic,
                             overload_method, overload, overload_attribute)
from numba.cpython.hashing import _Py_hash_t
from numba.core.imputils import (impl_ret_new_ref, impl_ret_borrowed, iternext_impl, RefType)
from numba.cpython.listobj import ListInstance
from numba.core.typing.templates import (infer_global, AbstractTemplate, infer,
                                         signature, AttributeTemplate, infer_getattr, bound_function)
from numba import prange

from sdc.str_ext import string_type
from sdc.str_arr_type import (StringArray, string_array_type, StringArrayType,
                              StringArrayPayloadType, str_arr_payload_type, StringArrayIterator,
                              is_str_arr_typ, offset_typ, data_ctypes_type, offset_ctypes_type)
from sdc.utilities.sdc_typing_utils import check_is_array_of_dtype

from sdc import hnative_dict

from sdc.utilities.sdc_typing_utils import sdc_pandas_index_types
from sdc.utilities.utils import sdc_overload
from sdc.datatypes.indexes import Int64IndexType

from sdc.extensions.sdc_stl_hashmap_type import SdcDictType
from sdc.extensions.sdc_stl_hashmap_ext import _get_types_postfixes, get_min_size, reduced_type_map, load_native_func


ll.add_symbol('hashmap_build_map_positions_int64_t', hnative_dict.hashmap_build_map_positions_int64_t)


@intrinsic
def build_map_positions(typingctx, keys):

    ty_key, ty_val = keys.dtype, types.int64
    dict_type = SdcDictType(ty_key, ty_val)
    key_type_postfix, _ = _get_types_postfixes(ty_key, ty_val)

    def codegen(context, builder, sig, args):
        keys_val, = args
        nrt_table = context.nrt.get_nrt_api(builder)

        keys_ctinfo = context.make_helper(builder, keys, keys_val)
        size_val = context.compile_internal(
            builder,
            lambda k: len(k),
            signature(types.int64, keys),
            [keys_val]
        )

        # create SdcDict struct and call native ctor filling meminfo
        lir_key_type = context.get_value_type(reduced_type_map.get(ty_key, ty_key))
        cdict = cgutils.create_struct_proxy(dict_type)(context, builder)
        fnty = lir.FunctionType(lir.VoidType(),
                                [cdict.meminfo.type.as_pointer(),    # meminfo to fill
                                 lir.IntType(8).as_pointer(),        # NRT API func table
                                 lir_key_type.as_pointer(),          # array of keys
                                 lir.IntType(64),                    # size
                                 ])
        func_name = f"hashmap_build_map_positions_{key_type_postfix}"
        fn_hashmap_create = builder.module.get_or_insert_function(
            fnty, name=func_name)
        builder.call(fn_hashmap_create,
                     [cdict._get_ptr_by_name('meminfo'),
                      nrt_table,
                      keys_ctinfo.data,
                      size_val
                      ])
        cdict.data_ptr = context.nrt.meminfo_data(builder, cdict.meminfo)
        return cdict._getvalue()

    return dict_type(keys), codegen


def sdc_indexes_map_positions(self):
    pass


@sdc_overload(sdc_indexes_map_positions)
def sdc_indexes_map_positions_ovld(self):

    if not isinstance(self, sdc_pandas_index_types):
        return None

    if isinstance(self, Int64IndexType):
        def sdc_indexes_map_positions_impl(self):
            map_last_pos = build_map_positions(self.values)
            return map_last_pos

        return sdc_indexes_map_positions_impl

