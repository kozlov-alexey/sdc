import numba
import hpat


def jit(signature_or_function=None, **options):
    # set nopython by default
    if 'nopython' not in options:
        options['nopython'] = True

    _locals = options.pop('locals', {})
    assert isinstance(_locals, dict)

    # put pivots in locals TODO: generalize numba.jit options
    pivots = options.pop('pivots', {})
    assert isinstance(pivots, dict)
    for var, vals in pivots.items():
        _locals[var+":pivot"] = vals

    h5_types = options.pop('h5_types', {})
    assert isinstance(h5_types, dict)
    for var, vals in h5_types.items():
        _locals[var+":h5_types"] = vals

    distributed = set(options.pop('distributed', set()))
    assert isinstance(distributed, (set, list))
    _locals["##distributed"] = distributed

    threaded = set(options.pop('threaded', set()))
    assert isinstance(threaded, (set, list))
    _locals["##threaded"] = threaded

    options['locals'] = _locals

    #options['parallel'] = True
    options['parallel'] = {'comprehension': True,
                           'setitem':       False,  # FIXME: support parallel setitem
                           'reduction':     True,
                           'numpy':         True,
                           'stencil':       True,
                           'fusion':        True,
                           }

    if hpat.isnotebook():
        print("isnotebook mode")
        import ipyparallel as ipp
#        c = ipp.Client('/home/taanders/.ipython/profile_default/security/ipcontroller-client.json', profile='mpi')
        c = ipp.Client('/home/taanders/.ipython/profile_mpi/security/ipcontroller-client.json', profile='mpi')
#        c = ipp.Client(profile='mpi')
        dview = c[:]
        if False:
            asres = dview.apply_sync(numba.jit, signature_or_function, pipeline_class=hpat.compiler.HPATPipeline, **options)
            print("asres", asres, type(asres))
            eres = dview.execute('1', block=True)
#            eres = c[0].execute('f2(1)', block=True)
            print("eres", eres, type(eres))
        else:
            @dview.remote(block=True)
            def numba_invoke():
                def f1():
                    return 1
                f1j = numba.jit(f1j, **options)
                return f1j()

    # this is for previous version of pipeline manipulation (numba hpat_req <0.38)
    # from .compiler import add_hpat_stages
    # return numba.jit(signature_or_function, user_pipeline_funcs=[add_hpat_stages], **options)
    return numba.jit(signature_or_function, pipeline_class=hpat.compiler.HPATPipeline, **options)
