#include <iostream>
#include <vector>
#include <algorithm>
#include <pthread.h>
#include <cmath>
#include <cstdlib>
#include <random>

//#define DEBUG_CHECK

// Task structure to define work for threads
struct Task {
    unsigned int* array;
    unsigned int step;
    unsigned int stage;
    unsigned int length;
    bool sort_increasing;
    unsigned int thread_id;  // Add thread ID to the task
};

// Shared index and mutex to manage dynamic work allocation
unsigned int shared_index = 0;
pthread_mutex_t index_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned int step_size = 1;
std::vector<unsigned int> thread_iterations;
pthread_mutex_t iterations_mutex = PTHREAD_MUTEX_INITIALIZER;

// Bitonic sort task for each thread
void* bitonic_sort_task(void* arg) {
    Task* task = static_cast<Task*>(arg);
    unsigned int* array = task->array;
    unsigned int step = task->step;
    unsigned int stage = task->stage;
    unsigned int length = task->length;
    bool sort_increasing = task->sort_increasing;
    unsigned int thread_id = task->thread_id;  // Retrieve thread ID

    const unsigned int same_order_block_width = 1 << step;
    const unsigned int pair_distance = 1 << (step - stage);
    const unsigned int sorted_block_width = 2 * pair_distance;
    int local_iterations = 0;

    while (true) {
        unsigned int start;
        local_iterations++;

        // Lock to get the next chunk of work
        pthread_mutex_lock(&index_mutex);
        start = shared_index;
        shared_index += step_size; // Each thread will process chunks of 10 pairs
        pthread_mutex_unlock(&index_mutex);

        if (start >= length / 2) {
            break; // No more work left
        }

        unsigned int end = std::min(start + step_size, length / 2);

        // Debug output with thread-safe mechanism (optional)
        // pthread_mutex_lock(&output_mutex);
        // std::cout << "Thread " << thread_id << " (pthread ID: " << pthread_self() << ") working on pairs from " << start << " to " << end << std::endl;
        // pthread_mutex_unlock(&output_mutex);

        // Process step_size pairs
        for (unsigned int id = start; id < end; ++id) {
            const unsigned int left_id = (id % pair_distance) + (id / pair_distance) * sorted_block_width;
            const unsigned int right_id = left_id + pair_distance;

            // Ensure no out-of-bounds access
            if (right_id >= length) {
                continue;
            }

            const unsigned int left_element = array[left_id];
            const unsigned int right_element = array[right_id];

            // Use a local copy of sort_increasing
            bool local_sort_increasing = sort_increasing;
            if ((id / same_order_block_width) % 2 == 1) {
                local_sort_increasing = !local_sort_increasing;
            }

            const unsigned int greater = (left_element > right_element) ? left_element : right_element;
            const unsigned int lesser = (left_element > right_element) ? right_element : left_element;
            array[left_id] = (local_sort_increasing) ? lesser : greater;
            array[right_id] = (local_sort_increasing) ? greater : lesser;
        }
    }

#ifdef DEBUG_CHECK
    pthread_mutex_lock(&iterations_mutex);
    thread_iterations[thread_id] = local_iterations;
    pthread_mutex_unlock(&iterations_mutex);
#endif

    delete task; // Free allocated memory for the task
    pthread_exit(nullptr);
}

int main(int argc, char* argv[]) {
    unsigned int steps = 15;  // 2^steps will be the length of the array
    bool sort_increasing = true;  // Sort in increasing order by default
    unsigned int num_threads = 2;  // Default number of threads

    if (argc > 1) {
        steps = std::stoi(argv[1]);  // User-defined steps
    }
    if (argc > 2) {
        std::string sort_order = argv[2];
        sort_increasing = (sort_order == "inc");
    }
    if (argc > 3) {
        num_threads = std::stoi(argv[3]);  // User-defined number of threads
    }

    unsigned int length = 1u << steps;
    std::vector<unsigned int> array(length);

    // Generate the array
    std::mt19937 rng(10);
    std::uniform_int_distribution<unsigned int> dist(0, 99);  // Uniform distribution in range [0, 99]
    std::for_each(array.begin(), array.end(), [&dist, &rng](unsigned int& e) {
        e = dist(rng);
    });

#ifdef DEBUG_CHECK
    std::vector<unsigned int> expected_array(array);  // For reference sorting
#endif

    std::cout << "Sorting an array of " << length << " elements using bitonic sort with " << num_threads << " threads." << std::endl;

    // Dynamically allocate resources based on number of threads
    pthread_t* threads = new pthread_t[num_threads];
    thread_iterations.resize(num_threads, 0);

    // Create tasks and assign them to threads for bitonic sort
    for (unsigned int i = 0; i < steps; ++i) {
        for (unsigned int j = 0; j <= i; ++j) {
            // Reset the shared index for the current step and stage
            shared_index = 0;

            // Create threads and let them work on chunks dynamically
            for (unsigned int t = 0; t < num_threads; ++t) {
                // Create a copy of the task for each thread to prevent memory issues
                Task* thread_task = new Task{array.data(), i, j, length, sort_increasing, t};
                pthread_create(&threads[t], nullptr, bitonic_sort_task, thread_task);
            }

            // Join all threads before proceeding to the next stage
            for (unsigned int t = 0; t < num_threads; ++t) {
                pthread_join(threads[t], nullptr);
            }
        }
    }

#ifdef DEBUG_CHECK
    // Verify results using reference implementation
    std::sort(expected_array.begin(), expected_array.end());
#endif

    // Print the number of iterations each thread completed
    std::cout << "Thread iteration counts:" << std::endl;
    for (unsigned int t = 0; t < num_threads; ++t) {
        std::cout << "Thread " << t << ": " << thread_iterations[t] << " iterations" << std::endl;
    }

#ifdef DEBUG_CHECK
    unsigned int errors = 0;
    for (unsigned int i = 0; i < length; ++i) {
        if (array[i] != expected_array[i]) {
            std::cout << "Mismatch at index " << i << ": expected " << expected_array[i] << ", got " << array[i] << std::endl;
            ++errors;
        }
    }

    if (errors == 0) {
        std::cout << "Sorting was successful!" << std::endl;
    } else {
        std::cout << "Errors detected in sorting. Number of errors: " << errors << std::endl;
    }
#endif 

    delete[] threads;  // Free the dynamically allocated array of threads

    return 0;
}
