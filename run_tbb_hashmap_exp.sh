#!/bin/bash

for n_threads in 1 2 4 8 16 28 56
do
    $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc -O2 -fPIC -isystem $CONDA_PREFIX/include -I$CONDA_PREFIX/include -std=c++11 -DTBB_PREVIEW_WAITING_FOR_WORKERS=1 -DCONFIGURE_NUM_THREADS=$n_threads -c ./sdc/native/test_hashmap_arena.cpp;
    $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ -Wl,-rpath-link,$CONDA_PREFIX/lib ./test_hashmap_arena.o -L$CONDA_PREFIX/lib -ltbb -o ./test_hashmap_arena -std=c++11;
    ./test_hashmap_arena;
    rm test_hashmap_arena test_hashmap_arena.o;
done

