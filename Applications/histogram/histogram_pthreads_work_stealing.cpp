#include <pthread.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <atomic>
#include <mutex>

//#define DEBUG_CHECK

// Define the structure for passing parameters to threads
struct ThreadData
{
    unsigned char* data;
    unsigned int* block_bins;
    int items_per_fetch;
    int num_threads;
    int bin_size;
    int size;
    std::atomic<int>* current_index;
    std::mutex* index_mutex;
    int* chunks_processed; // Added to track chunks processed by each thread
    int thread_index; // Added to keep track of the thread's index
};

// Thread function to compute histogram for each thread's portion of the data
void* histogram256_block(void* arg)
{
    ThreadData* td = (ThreadData*)arg;
    unsigned char* data = td->data;
    unsigned int* block_bins = td->block_bins;
    int items_per_fetch = td->items_per_fetch;
    int num_threads = td->num_threads;
    int bin_size = td->bin_size;
    int size = td->size;
    std::atomic<int>* current_index = td->current_index;
    std::mutex* index_mutex = td->index_mutex;
    int* chunks_processed = td->chunks_processed; // Get the pointer to chunks_processed
    int thread_index = td->thread_index; // Get the thread index

    // Initialize thread_bins for each thread
    std::vector<unsigned int> thread_bins(bin_size, 0);

    int chunks_count = 0;

    while (true)
    {
        int start_index;

        // Lock and get the next starting index for processing
        index_mutex->lock();
        start_index = *current_index;
        *current_index += items_per_fetch;
        index_mutex->unlock();

        // Exit condition if no more data to process
        if (start_index >= size)
        {
            break;
        }

        // Process the allocated chunk of data
        int end_index = std::min(start_index + items_per_fetch, size);
        for (int i = start_index; i < end_index; ++i)
        {
            unsigned int value = data[i];
            thread_bins[value]++;
        }

        chunks_count++;
    }

#ifdef DEBUG_CHECK
    // Update chunks processed for this thread
    chunks_processed[thread_index] = chunks_count;
#endif

    // Accumulate bins across all threads
    for (int i = 0; i < bin_size; ++i)
    {
        block_bins[thread_index * bin_size + i] = thread_bins[i];
    }

    return nullptr;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <size>" << std::endl;
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    int size = std::stoi(argv[2]);
    const int items_per_fetch = 10;
    const int bin_size = 256;

    // Fixed seed for reproducibility
    const unsigned int seed = 12345;

    std::vector<unsigned char> h_data(size);
    std::vector<unsigned int> h_bins(bin_size, 0);

    // Randomly generate input data with a fixed seed
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<unsigned int> distribution(0, 255);
    std::generate(h_data.begin(), h_data.end(), [&]() { return distribution(generator); });

    std::vector<unsigned int> block_bins(num_threads * bin_size, 0);
    std::vector<int> chunks_processed(num_threads, 0);
    std::atomic<int> current_index(0);
    std::mutex index_mutex;

    // Prepare thread data
    std::vector<ThreadData> thread_data(num_threads);
    std::vector<pthread_t> threads(num_threads);

    for (int i = 0; i < num_threads; ++i)
    {
        thread_data[i] = {h_data.data(), block_bins.data(), items_per_fetch, num_threads, bin_size, size, &current_index, &index_mutex, chunks_processed.data(), i};
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

#ifdef DEBUG_CHECK
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

    // Print out how many chunks each thread processed
    std::cout << "Chunks processed by each thread:" << std::endl;
    for (int i = 0; i < num_threads; ++i)
    {
        std::cout << "Thread " << i << ": " << chunks_processed[i] << " chunks" << std::endl;
    }
#endif
    return 0;
}
