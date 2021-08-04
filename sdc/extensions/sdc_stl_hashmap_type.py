# *****************************************************************************
# Copyright (c) 2021, Intel Corporation All rights reserved.
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

from numba.core.typing.templates import (
    infer_global, AbstractTemplate, signature,
    )
from numba.extending import type_callable, lower_builtin
from numba import types
from numba.extending import (models, register_model, make_attribute_wrapper, overload_method)
from sdc.str_ext import string_type

from collections.abc import MutableMapping
from numba.core.types import Dummy, IterableType, SimpleIterableType, SimpleIteratorType

from numba.extending import typeof_impl
from numba.core.typing.typeof import _typeof_type as numba_typeof_type


class SdcDictIteratorType(SimpleIteratorType):
    def __init__(self, iterable):
        self.parent = iterable.parent
        self.iterable = iterable
        yield_type = iterable.yield_type
        name = "iter[{}->{}],{}".format(
            iterable.parent, yield_type, iterable.name
        )
        super(SdcDictIteratorType, self).__init__(name, yield_type)


class SdcDictKeysIterableType(SimpleIterableType):
    """Concurrent SdcDictionary iterable type for .keys()
    """

    def __init__(self, parent):
        assert isinstance(parent, SdcDictType)
        self.parent = parent
        self.yield_type = self.parent.key_type
        name = "keys[{}]".format(self.parent.name)
        self.name = name
        iterator_type = SdcDictIteratorType(self)
        super(SdcDictKeysIterableType, self).__init__(name, iterator_type)


class SdcDictItemsIterableType(SimpleIterableType):
    """Concurrent SdcDictionary iterable type for .items()
    """

    def __init__(self, parent):
        assert isinstance(parent, SdcDictType)
        self.parent = parent
        self.yield_type = self.parent.keyvalue_type
        name = "items[{}]".format(self.parent.name)
        self.name = name
        iterator_type = SdcDictIteratorType(self)
        super(SdcDictItemsIterableType, self).__init__(name, iterator_type)


class SdcDictValuesIterableType(SimpleIterableType):
    """Concurrent SdcDictionary iterable type for .values()
    """

    def __init__(self, parent):
        assert isinstance(parent, SdcDictType)
        self.parent = parent
        self.yield_type = self.parent.value_type
        name = "values[{}]".format(self.parent.name)
        self.name = name
        iterator_type = SdcDictIteratorType(self)
        super(SdcDictValuesIterableType, self).__init__(name, iterator_type)


@register_model(SdcDictItemsIterableType)
@register_model(SdcDictKeysIterableType)
@register_model(SdcDictValuesIterableType)
@register_model(SdcDictIteratorType)
class SdcDictIterModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('parent', fe_type.parent),  # reference to the dict
            ('state', types.voidptr),    # iterator state in native code
            ('meminfo', types.MemInfoPointer(types.voidptr)),   # meminfo for allocated iter state
        ]
        super(SdcDictIterModel, self).__init__(dmm, fe_type, members)


class SdcDictType(IterableType):
    def __init__(self, keyty, valty):
        self.key_type = keyty
        self.value_type = valty
        self.keyvalue_type = types.Tuple([keyty, valty])
        super(SdcDictType, self).__init__(
            name='SdcDictType({}, {})'.format(keyty, valty))

    @property
    def iterator_type(self):
        return SdcDictKeysIterableType(self).iterator_type


@register_model(SdcDictType)
class SdcDictModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        members = [
            ('data_ptr', types.CPointer(types.uint8)),
            ('meminfo', types.MemInfoPointer(types.voidptr)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(SdcDictType, 'data_ptr', '_data_ptr')


class SdcDict(MutableMapping):
    def __new__(cls, dcttype=None, meminfo=None):
        return object.__new__(cls)

    @classmethod
    def empty(cls, key_type, value_type):
        return cls(dcttype=SdcDictType(key_type, value_type))

    @classmethod
    def from_arrays(cls, keys, values):
        return cls(dcttype=SdcDictType(keys.dtype, values.dtype))

    @classmethod
    def fromkeys(cls, keys, value):
        return cls(dcttype=SdcDictType(keys.dtype, value))

    def __init__(self, **kwargs):
        if kwargs:
            self._dict_type, self._opaque = self._parse_arg(**kwargs)
        else:
            self._dict_type = None

    @property
    def _numba_type_(self):
        if self._dict_type is None:
            raise TypeError("invalid operation on untyped dictionary")
        return self._dict_type


# FIXME_Numba#6781: due to overlapping of overload_methods for Numba TypeRef
# we have to use our new SdcTypeRef to type objects created from types.Type
# (i.e. SdcDict meta-type). This should be removed once it's fixed.
class SdcTypeRef(Dummy):
    """Reference to a type.

    Used when a type is passed as a value.
    """
    def __init__(self, instance_type):
        self.instance_type = instance_type
        super(SdcTypeRef, self).__init__('sdc_typeref[{}]'.format(self.instance_type))


@register_model(SdcTypeRef)
class SdcTypeRefModel(models.OpaqueModel):
    def __init__(self, dmm, fe_type):

        models.OpaqueModel.__init__(self, dmm, fe_type)


@typeof_impl.register(type)
def mynew_typeof_type(val, c):
    """ This function is a workaround for """

    if not issubclass(val, SdcDict):
        return numba_typeof_type(val, c)
    else:
        return SdcTypeRef(SdcDictType)
