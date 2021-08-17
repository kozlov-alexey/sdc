@echo off
setlocal enabledelayedexpansion

FOR %%x in (1, 2, 4, 8, ) do (
    set NUMBA_NUM_THREADS=%%x
    echo Running with n_threads !NUMBA_NUM_THREADS!
    python -W ignore test_pandas_get_indexer.py
)