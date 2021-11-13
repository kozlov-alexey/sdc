#include <chrono>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <random>

#include "tbb/tbb.h"
#include "tbb/task_arena.h"
#include "libcuckoo/cuckoohash_map.hh"

#ifndef CONFIGURE_NUM_THREADS
#define CONFIGURE_NUM_THREADS 1
#endif

int64_t PROBLEM_SIZE = 10000000;
int64_t MAX_SIZE = 10000000;

using namespace std;
using namespace std::chrono;

using conc_hash_map_int64_pos = libcuckoo::cuckoohash_map<int64_t, int64_t>;

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int64_t> data;
    data.reserve(PROBLEM_SIZE);
    for (int i; i < PROBLEM_SIZE; i++)
        data.push_back(i);
    std::shuffle(data.begin(), data.end(), gen);

    size_t res = 0;
    // std::vector<int> n_threads_vec = {1, 2, 4, 8, 16, 28, 56};
    // std::vector<int> n_threads_vec = {8};
    // std::vector<int> n_threads_vec = {4};
    // std::vector<int> n_threads_vec = {1};
    // std::vector<int> n_threads_vec = {28};
    std::vector<int> n_threads_vec = {CONFIGURE_NUM_THREADS};
    for (auto n_threads : n_threads_vec) {
        tbb::task_arena tbb_arena(n_threads);

        auto t1 = high_resolution_clock::now();
        auto size = PROBLEM_SIZE;
        conc_hash_map_int64_pos hash_table(size);
        auto ptr_my_map = &hash_table;
        tbb_arena.execute([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, size),
                [&](const tbb::blocked_range<size_t>& r) {
                    for(size_t i=r.begin(); i!=r.end(); ++i) {
                        ptr_my_map->insert(data[i], i);
                    }
                }
            );
        });
        res = hash_table.size();
        auto t2 = high_resolution_clock::now();
        duration<double, std::ratio<1, 1>> ms_double = t2 - t1;
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        std::cout << "Creating map_positions, nthreads=" << n_threads << " took: " << ms_int.count() << " ms, (" << ms_double.count() << " sec)"
                  << std::endl;
    }
    std::cout << "result map size: "  << res << std::endl;
    return 0;
}

#undef CONFIGURE_NUM_THREADS