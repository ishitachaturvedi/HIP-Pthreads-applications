#include <pthread.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

// Define the structure for passing parameters to threads
struct ThreadData
{
    unsigned char* data;
    unsigned int* block_bins;
    int items_per_thread;
    int thread_id;
    int threads_per_block;
    int bin_size;
    int size;
};

// Thread function to compute histogram for each thread's portion of the data
void* histogram256_block(void* arg)
{
    ThreadData* td = (ThreadData*)arg;
    unsigned char* data = td->data;
    unsigned int* block_bins = td->block_bins;
    int items_per_thread = td->items_per_thread;
    int thread_id = td->thread_id;
    int threads_per_block = td->threads_per_block;
    int bin_size = td->bin_size;
    int size = td->size;

    // Initialize thread_bins for each thread
    std::vector<unsigned int> thread_bins(bin_size, 0);

    for (int i = 0; i < items_per_thread && (thread_id * items_per_thread + i) < size; ++i)
    {
        unsigned int value = data[thread_id * items_per_thread + i];
        thread_bins[value]++;
    }

    // Accumulate bins across all threads
    for (int i = 0; i < bin_size; ++i)
    {
        block_bins[thread_id * bin_size + i] = thread_bins[i];
    }

    return nullptr;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <size> <items_per_thread>" << std::endl;
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    int size = std::stoi(argv[2]);
    int items_per_thread = std::stoi(argv[3]);

    const int bin_size = 256;

    std::vector<unsigned char> h_data(size);
    std::vector<unsigned int> h_bins(bin_size, 0);

    // Randomly generate input data
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(0, 255);
    std::generate(h_data.begin(), h_data.end(), [&]() { return distribution(generator); });

    std::vector<unsigned int> block_bins(num_threads * bin_size, 0);

    // Prepare thread data
    std::vector<ThreadData> thread_data(num_threads);
    std::vector<pthread_t> threads(num_threads);

    for (int i = 0; i < num_threads; ++i)
    {
        thread_data[i] = {h_data.data(), block_bins.data(), items_per_thread, i, num_threads, bin_size, size};
        pthread_create(&threads[i], nullptr, histogram256_block, &thread_data[i]);
    }

    // Join threads
    for (int i = 0; i < num_threads; ++i)
    {
        pthread_join(threads[i], nullptr);
    }

    // Combine results from each thread
    for (int i = 0; i < num_threads; ++i)
    {
        for (int j = 0; j < bin_size; ++j)
        {
            h_bins[j] += block_bins[i * bin_size + j];
        }
    }

    // Verification (calculate reference histogram)
    std::vector<unsigned int> reference_bins(bin_size, 0);
    for (int i = 0; i < size; ++i)
    {
        reference_bins[h_data[i]]++;
    }

    // Check results
    int errors = 0;
    for (int i = 0; i < bin_size; ++i)
    {
        if (h_bins[i] != reference_bins[i])
        {
            errors++;
        }
    }

    if (errors == 0)
    {
        std::cout << "Validation passed." << std::endl;
    }
    else
    {
        std::cout << "Validation failed with " << errors << " errors." << std::endl;
    }

    return 0;
}
