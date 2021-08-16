#include <chrono>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <random>

// #pragma GCC push_options
// #pragma GCC optimize ("-O0")
#include "tbb/tbb.h"
#include "tbb/task_arena.h"
// #pragma GCC pop_options
// #include "hashmap.hpp"

int64_t PROBLEM_SIZE = 10000000;
int64_t MAX_SIZE = 10000000;

using namespace std;
using namespace std::chrono;

class TrivialTBBHashCompare {
public:
    static size_t hash(const int64_t& val) {
        return (size_t)val;
    }
    static bool equal(const int64_t& k1, const int64_t& k2) {
        return k1==k2;
    }
};

class TrivialHash
{
public:
    size_t operator()(const int64_t& val) const {
        return (size_t)val;
    }
};

using conc_hash_map_int64_pos = tbb::concurrent_hash_map<int64_t, int64_t, TrivialTBBHashCompare>;
//using conc_hash_map_int64_pos = tbb::concurrent_hash_map<int64_t, int64_t>;
//using conc_hash_map_int64_pos = tbb::concurrent_unordered_map<int64_t, int64_t, TrivialHash>;

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
    std::vector<int> n_threads_vec = {4};
    // std::vector<int> n_threads_vec = {1};
    for (auto n_threads : n_threads_vec) {
        tbb::task_arena tbb_arena(n_threads);

        auto t1 = high_resolution_clock::now();
        auto size = PROBLEM_SIZE;
        conc_hash_map_int64_pos tbb_map(size, TrivialTBBHashCompare());
        auto ptr_my_map = &tbb_map;
        tbb_arena.execute([&]() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, size),
                [&](const tbb::blocked_range<size_t>& r) {
                    for(size_t i=r.begin(); i!=r.end(); ++i) {
                        ptr_my_map->emplace(data[i], i);
                    }
                }
            );
        });
        res = tbb_map.size();
        auto t2 = high_resolution_clock::now();
        duration<double, std::ratio<1, 1>> ms_double = t2 - t1;
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        std::cout << "Creating map_positions, nthreads=" << n_threads << " took: " << ms_int.count() << " ms, (" << ms_double.count() << " sec)"
                  << std::endl;
    }
    std::cout << "result map size: "  << res << std::endl;
    return 0;
}
