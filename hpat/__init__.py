from __future__ import print_function, division, absolute_import

import numba
from numba import *
import hpat.dict_ext
import hpat.set_ext
from hpat.set_ext import init_set_string
import hpat.distributed_api
from hpat.distributed_api import dist_time
from hpat.dict_ext import DictIntInt, DictInt32Int32, dict_int_int_type, dict_int32_int32_type
from hpat.str_ext import string_type
from hpat.str_arr_ext import string_array_type
from numba.types import List
from hpat.utils import cprint, distribution_report_from_analysis
import hpat.compiler
import hpat.io
import hpat.pd_timestamp_ext
import hpat.config
import hpat.timsort
import copy

if hpat.config._has_xenon:
    from hpat.xenon_ext import read_xenon, xe_connect, xe_open, xe_close

multithread_mode = False


def jit(signature_or_function=None, **options):
    # set nopython by default
    if 'nopython' not in options:
        options['nopython'] = True

    _locals = options.pop('locals', {})

    # put pivots in locals TODO: generalize numba.jit options
    pivots = options.pop('pivots', {})
    for var, vals in pivots.items():
        _locals[var+":pivot"] = vals

    options['locals'] = _locals

    #options['parallel'] = True
    options['parallel'] = {'comprehension': True,
                           'setitem':       False,  # FIXME: support parallel setitem
                           'reduction':     True,
                           'numpy':         True,
                           'stencil':       True,
                           'fusion':        True,
                           }

    # this is for previous version of pipeline manipulation (numba hpat_req <0.38)
    # from .compiler import add_hpat_stages
    # return numba.jit(signature_or_function, user_pipeline_funcs=[add_hpat_stages], **options)

    # Get the dispatcher object from Numba.
    orig_dispatcher = numba.jit(signature_or_function, pipeline_class=hpat.compiler.HPATPipeline, **options)
    # Get the type of the dispatcher.
    tod = type(orig_dispatcher)
    # Create a HPAT dispatcher type that derives from the dispatcher type returned by Numba.
    # This allows us to add additional methods to operate on the dispatcher.
    class hpat_dispatcher(tod):
        def distribution_report(self, signature=None, level=1):
            self.parallel_diagnostics(level=level) 
            if not hasattr(self, 'sig_to_cres'):
                return
            if signature is not None:
                distribution_report_from_analysis(self.sig_to_cres[signature][1], fccres=(self.sig_to_cres[sig][2], self.sig_to_cres[sig][0]))
            else:
                [distribution_report_from_analysis(self.sig_to_cres[sig][1], fccres=(self.sig_to_cres[sig][2], self.sig_to_cres[sig][0])) for sig in self.signatures]

        def compile(self, sig):
            if not hasattr(self, 'sig_to_cres'):
                self.sig_to_cres = {}
            super(hpat_dispatcher, self).compile(sig)

            args, return_typ = numba.sigutils.normalize_signature(sig)
            cres = self.overloads.get(tuple(args))
            self.sig_to_cres[sig] = (cres, copy.copy(distributed.dist_analysis), compiler.last_ir)
            return cres.entry_point

    # CRAZY! Change the type of the dispatcher to be our new derived type.
    orig_dispatcher.__class__ = hpat_dispatcher

    return orig_dispatcher
