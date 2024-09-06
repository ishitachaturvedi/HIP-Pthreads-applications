#include <iostream>
#include <vector>
#include <algorithm>
#include <pthread.h>
#include <cstdlib>

// Task structure to define work for threads
struct Task {
    unsigned int* array;
    unsigned int step;
    unsigned int stage;
    unsigned int length;
    bool sort_increasing;
    unsigned int start_idx; // Start index of the chunk to sort
    unsigned int chunk_size; // Size of the chunk to sort
};

// Bitonic sort task for each thread
void* bitonic_sort_task(void* arg) {
    Task* task = static_cast<Task*>(arg);
    unsigned int* array = task->array;
    unsigned int step = task->step;
    unsigned int stage = task->stage;
    unsigned int length = task->length;
    bool sort_increasing = task->sort_increasing;
    unsigned int start_idx = task->start_idx;
    unsigned int chunk_size = task->chunk_size;

    const unsigned int same_order_block_width = 1 << step;
    const unsigned int pair_distance = 1 << (step - stage);
    const unsigned int sorted_block_width = 2 * pair_distance;

    for (unsigned int thread_id = start_idx; thread_id < start_idx + chunk_size; ++thread_id) {
        const unsigned int left_id = (thread_id % pair_distance) + (thread_id / pair_distance) * sorted_block_width;
        const unsigned int right_id = left_id + pair_distance;

        // Check boundaries before accessing array elements
        if (right_id >= length) {
            continue; // Skip processing if right_id is out of bounds
        }

        const unsigned int left_element = array[left_id];
        const unsigned int right_element = array[right_id];

        // Determine the sorting order for this chunk
        bool local_sort_increasing = sort_increasing;
        if ((thread_id / same_order_block_width) % 2 == 1) {
            local_sort_increasing = !local_sort_increasing;
        }

        const unsigned int greater = (left_element > right_element) ? left_element : right_element;
        const unsigned int lesser = (left_element > right_element) ? right_element : left_element;
        array[left_id] = (local_sort_increasing) ? lesser : greater;
        array[right_id] = (local_sort_increasing) ? greater : lesser;
    }

    pthread_exit(nullptr);

}

// Reference CPU implementation of the bitonic sort for result verification
void bitonic_sort_reference(unsigned int* array, unsigned int length, bool sort_increasing) {
    const unsigned int half_length = length / 2;

    for (unsigned int i = 2; i <= length; i *= 2) {
        for (unsigned int j = i; j > 1; j /= 2) {
            bool increasing = sort_increasing;
            const unsigned int half_j = j / 2;

            for (unsigned int k = 0; k < length; k += j) {
                const unsigned int k_plus_half_j = k + half_j;

                if ((k == i) || ((i < length) && (k % i) == 0 && (k != half_length))) {
                    increasing = !increasing;
                }

                for (unsigned int l = k; l < k_plus_half_j; ++l) {
                    if (increasing) {
                        if (array[l] > array[l + half_j]) {
                            std::swap(array[l], array[l + half_j]);
                        }
                    } else {
                        if (array[l + half_j] > array[l]) {
                            std::swap(array[l + half_j], array[l]);
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    unsigned int steps = 15;
    bool sort_increasing = true;

    if (argc > 1) {
        steps = std::stoi(argv[1]);
    }
    if (argc > 2) {
        std::string sort_order = argv[2];
        sort_increasing = (sort_order == "inc");
    }

    unsigned int length = 1u << steps;
    std::vector<unsigned int> array(length);
    std::for_each(array.begin(), array.end(), [](unsigned int& e) { e = rand() % 100; });
    std::vector<unsigned int> expected_array(array);

    std::cout << "Sorting an array of " << length << " elements using bitonic sort." << std::endl;

    // Pthread setup
    unsigned int num_threads = 1;
    pthread_t threads[num_threads];

    // Bitonic sort: launch threads for each stage of each step
    for (unsigned int i = 0; i < steps; ++i) {
        for (unsigned int j = 0; j <= i; ++j) {
            unsigned int chunk_size = length / num_threads;

            for (unsigned int t = 0; t < num_threads; ++t) {
                Task* task = new Task{array.data(), i, j, length, sort_increasing, t * chunk_size, chunk_size};
                pthread_create(&threads[t], nullptr, bitonic_sort_task, task);
            }

            // Join all threads
            for (unsigned int t = 0; t < num_threads; ++t) {
                pthread_join(threads[t], nullptr);
            }
        }
    }

    // Execute CPU bitonic sort for reference
    bitonic_sort_reference(expected_array.data(), length, sort_increasing);

    // Validate results
    unsigned int errors = 0;
    for (unsigned int i = 0; i < length; ++i) {
        if (array[i] != expected_array[i]) {
            ++errors;
        }
    }

    if (errors == 0) {
        std::cout << "Sorting was successful!" << std::endl;
    } else {
        std::cout << "Errors detected in sorting. Number of errors: " << errors << std::endl;
    }

    return 0;
}
