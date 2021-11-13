#!/bin/bash

for n_threads in 1 2 4 8 16 28 56
do
    $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc -O2 -fPIC -isystem $CONDA_PREFIX/include -I$CONDA_PREFIX/include -I$SDC_DIR/../libcuckoo/install/include/ -std=c++11 -DTBB_PREVIEW_WAITING_FOR_WORKERS=1 -DCONFIGURE_NUM_THREADS=$n_threads -c ./sdc/native/test_cuckoo_hash.cpp;
    $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ -Wl,-rpath-link,$CONDA_PREFIX/lib ./test_cuckoo_hash.o -L$CONDA_PREFIX/lib -ltbb -o ./test_cuckoo_hash -std=c++11;
    ./test_cuckoo_hash;
    rm test_cuckoo_hash test_cuckoo_hash.o;
done

