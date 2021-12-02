#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <math.h>

#ifdef _DEBUG
#undef _ITERATOR_DEBUG_LEVEL
#endif

#include <unordered_map>

#ifdef _DEBUG
#define _ITERATOR_DEBUG_LEVEL 2
#endif

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

size_t trivial_hash(const int64_t& val)
{
    return (size_t)val;
}


struct TrivialHash {
public:
    TrivialHash() = default;
    TrivialHash(const TrivialHash&) = default;
    ~TrivialHash() = default;
    size_t operator()(const int64_t& val) const {
        return (size_t)val;
    }

};

// using conc_hash_map_int64_pos = tbb::concurrent_hash_map<int64_t, int64_t, TrivialTBBHashCompare>;
//using conc_hash_map_int64_pos = tbb::concurrent_hash_map<int64_t, int64_t>;

void test_map(size_t size, int n_threads)
{

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int64_t> data;
    data.reserve(PROBLEM_SIZE);
    for (int i; i < PROBLEM_SIZE; i++)
        data.push_back(i);
    std::shuffle(data.begin(), data.end(), gen);

    auto t1 = high_resolution_clock::now();
    TrivialHash trivial_hash;
    std::unordered_map<int64_t, int64_t, TrivialHash> my_map(size, trivial_hash);
    for (int i=0; i < size; ++i)
    {
        my_map.emplace(data[i], i);
    }

    auto t2 = high_resolution_clock::now();
    duration<double, std::ratio<1, 1>> ms_double = t2 - t1;
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "Creating map_positions, nthreads=" << n_threads << " took: " << ms_int.count() << " ms, (" << ms_double.count() << " sec)"
                  << std::endl;
}

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());

//    std::vector<int> n_threads_vec = {1, 2, 4, 8, 16, 28, 56};
     std::vector<int> n_threads_vec = {1};
    // std::vector<int> n_threads_vec = {8};
    // std::vector<int> n_threads_vec = {28};
    // std::vector<int> n_threads_vec = {4};
    // std::vector<int> n_threads_vec = {2};
    for (auto n_threads : n_threads_vec) {
        test_map(PROBLEM_SIZE, n_threads);
    }

    return 0;
}

