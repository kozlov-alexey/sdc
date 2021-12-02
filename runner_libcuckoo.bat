@echo off
setlocal enabledelayedexpansion

FOR %%x in (1, 2, 4, 8, ) do (
    echo Running with n_threads %%x
    START /WAIT cl /c /nologo  /GL /GR /Ox /GL /W3 /DWIN32 /D_WINDOWS /DNDEBUG /MD -I"C:\Users\akozlov\AppData\Local\Continuum\anaconda3\libcuckoo" -I"%CONDA_PREFIX%\Library\include" -I"%CONDA_PREFIX%\include" -I"C:\Program Files (x86)\Windows Kits\10\include\10.0.17763.0\ucrt" -I"C:\Program Files (x86)\Windows Kits\10\include\10.0.17763.0\shared" -I"C:\Program Files (x86)\Windows Kits\10\include\10.0.17763.0\um" -I"C:\Program Files (x86)\Windows Kits\10\include\10.0.17763.0\winrt" /EHsc .\sdc\native\test_cuckoo_hash.cpp /DTBB_PREVIEW_WAITING_FOR_WORKERS=1 /DCONFIGURE_NUM_THREADS=%%x
    START /WAIT link /nologo /INCREMENTAL:NO /LTCG /MANIFEST:EMBED,ID=2 /MANIFESTUAC:NO  /LIBPATH:"%CONDA_PREFIX%\Lib" tbb.lib test_cuckoo_hash.obj
    .\test_cuckoo_hash.exe
)
