#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iterator>
#include <iostream>
#include <cstdio>

//#define DEBUG_CHECK

// Global variables
int global_index = 0;  // Shared global index for dynamic work assignment
pthread_mutex_t lock;  // Mutex lock for protecting global index

typedef struct {
    float* input;
    float* output;
    int size;
    int* block_sum;  // To store the sum of each block for global adjustment
    int num_threads;
} thread_data_t;

// Function for fetching the next chunk of work
int fetch_next_chunk(int* start_idx, int chunk_size, int total_size) {
    int end_idx;
    pthread_mutex_lock(&lock);
    *start_idx = global_index;
    global_index += chunk_size;
    pthread_mutex_unlock(&lock);

    end_idx = *start_idx + chunk_size;
    if (end_idx > total_size) {
        end_idx = total_size;  // Clamp the end_idx to the array size
    }
    return end_idx;
}

// Function for block-level prefix sum with dynamic work fetching
void* dynamic_prefix_sum(void* arg) {
    thread_data_t* t_data = (thread_data_t*)arg;
    float* input = t_data->input;
    float* output = t_data->output;
    int size = t_data->size;
    int chunk_size = 10;  // Process 10 elements at a time

    int start_idx, end_idx;

    while ((end_idx = fetch_next_chunk(&start_idx, chunk_size, size)) > start_idx) {
        // Compute the local prefix sum for this chunk
        if (start_idx > 0) {
            output[start_idx] = input[start_idx] + output[start_idx - 1];
        } else {
            output[start_idx] = input[start_idx];
        }

        for (int i = start_idx + 1; i < end_idx; i++) {
            output[i] = output[i - 1] + input[i];
        }
    }

    pthread_exit(NULL);
}

// Function to run prefix sum using pthreads with dynamic work allocation
void run_dynamic_prefix_sum(float* input, float* output, int size, int num_threads) {
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    thread_data_t* thread_data = (thread_data_t*)malloc(num_threads * sizeof(thread_data_t));

    pthread_mutex_init(&lock, NULL);  // Initialize the mutex lock

    // Set thread data
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input = input;
        thread_data[i].output = output;
        thread_data[i].size = size;
        thread_data[i].num_threads = num_threads;
        pthread_create(&threads[i], NULL, dynamic_prefix_sum, (void*)&thread_data[i]);
    }

    // Join threads after processing
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Free dynamically allocated memory and destroy mutex
    free(threads);
    free(thread_data);
    pthread_mutex_destroy(&lock);
}

int main(int argc, char* argv[]) {
    // Parse the input
    int size = 16;
    int num_threads = 4; // Default to 4 threads if not provided
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    if (argc > 2) {
        num_threads = atoi(argv[2]);
    }

    if (size <= 0 || num_threads <= 0) {
        printf("Size and num_threads must be at least 1.\n");
        return -1;
    }

    // Allocate memory for input and output
    float* input = (float*)malloc(sizeof(float) * size);
    float* output = (float*)malloc(sizeof(float) * size);

    // Fixed integer seed for reproducibility
    const unsigned int seed = 42;  // Use an integer seed

    // Randomly generate input data with a fixed seed
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    //for (int i = 0; i < size; i++) {
    //    input[i] = distribution(generator);
    //    printf("input[%d] = %f\n", i, input[i]);
    //}

    // Run the prefix sum algorithm using pthreads with dynamic work allocation
    run_dynamic_prefix_sum(input, output, size, num_threads);
#ifdef DEBUG_CHECK

     // 4. Verify the output.
    float verify = 0;
    int   errors = 0;
    for(int i = 0; i < size; i++)
    {
        verify += input[i];
        errors += std::pow(output[i] - verify, 2) > 1e-8;
    }

    std::cout << "Final sum on \n"
              << "  device: " << output[size -1] << "\n"
              << "  host  : " << verify << "\n"
              << std::endl;

    // Print the input and output arrays
    // printf("Input array:  ");
    // for (int i = 0; i < size; i++) {
    //     printf("%d ", input[i]);
    // }
    // printf("\nOutput array: ");
    // for (int i = 0; i < size; i++) {
    //     printf("%d ", output[i]);
    // }
    // printf("\n");

#endif
    // Clean up memory
    free(input);
    free(output);

    return 0;
}
