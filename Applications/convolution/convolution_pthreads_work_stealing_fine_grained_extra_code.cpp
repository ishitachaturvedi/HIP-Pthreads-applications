#include <iostream>
#include <vector>
#include <random>
#include <pthread.h>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cstdlib>  // for std::atoi
#include <atomic>    // for std::atomic

//#define DEBUG_CHECK

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t barrier;  // Global barrier
unsigned int current_column = 0;  // Shared variable for column work-stealing
const unsigned int columns_per_batch = 10;  // Number of columns each thread processes at a time

// Thread data structure
struct ThreadData {
    const float* input;
    float* output;
    const float* mask;
    unsigned int width;
    unsigned int padded_width;
    unsigned int mask_width;
    unsigned int height;
    unsigned int thread_id;
    std::atomic<unsigned int> columns_processed;  // Atomic counter for columns processed
};

// Function for performing convolution with work-stealing on the columns
void* convolution_thread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    
    const unsigned int width = data->width;
    const unsigned int padded_width = data->padded_width;
    const unsigned int mask_width = data->mask_width;
    const float* input = data->input;
    float* output = data->output;
    const float* mask = data->mask;
    const unsigned int height = data->height;

    // Process each row
    for (unsigned int y = 0; y < height; ++y) {
        bool has_work = true;
        unsigned int start_column = 0;
        unsigned int end_column = 0;
        current_column = 0;
        while (has_work) {
            // Work stealing: Lock mutex to safely update the shared column counter
            pthread_mutex_lock(&mutex);
            start_column = current_column;
            current_column += columns_per_batch;
            if (start_column < width) {
                end_column = std::min(start_column + columns_per_batch, width);
                has_work = true;
            } else {
                has_work = false;
            }
            pthread_mutex_unlock(&mutex);

            if (has_work) {
                for (unsigned int x = start_column; x < end_column; ++x) {
                    // Perform convolution on the current column for the given row
                    float sum = 0.0f;  // Local sum for each thread
                    for (unsigned int mask_index_y = 0; mask_index_y < mask_width; ++mask_index_y) {
                        for (unsigned int mask_index_x = 0; mask_index_x < mask_width; ++mask_index_x) {
                            unsigned int mask_index = mask_index_y * mask_width + mask_index_x;
                            unsigned int input_index = (y + mask_index_y) * padded_width + (x + mask_index_x);
                            sum += input[input_index] * mask[mask_index];

                            // Additional computational load: perform redundant operations
                            float temp = sum * 0.12345f; // Arbitrary multiplication
                            temp = temp * temp;           // Squaring the value
                            temp = temp + sum;            // Adding it back to sum
                            temp = temp * 1.6789f;        // More multiplications
                            temp = temp / (sum + 0.00001f);  // Division to make it more intensive
                            temp = sinf(temp);            // Sin function
                            temp = cosf(temp);            // Cos function
                            temp = expf(temp);            // Exponential function
                            temp = sqrtf(temp);           // Square root for additional complexity
                            sum += temp;                  // Accumulate result back into sum

                            // Repeat similar operations to further increase computational load
                            float temp2 = (sum + 1.2345f) * 2.3456f; // Arbitrary multiplication
                            temp2 = logf(fabsf(temp2 + 0.00001f));   // Logarithm for additional complexity
                            temp2 = powf(temp2, 2.0f);               // Raise to a power
                            sum += temp2;                            // Accumulate result back into sum
                        }
                    }
                    output[(y * width + x)] = sum;
                }
#ifdef DEBUG_CHECK
                // Increment the column counter for this thread
                data->columns_processed++;
#endif
            }

            // Synchronize threads if no more work is available
            if (!has_work) {
                pthread_barrier_wait(&barrier);
            }
        }

        // Wait for all threads to synchronize before starting the next row
        pthread_barrier_wait(&barrier);
    }

    return nullptr;
}


// Single-threaded convolution function (for validation)
void convolution_single_threaded(const float* input, float* output, const float* mask,
                                 unsigned int width, unsigned int padded_width, unsigned int mask_width, unsigned int height) {
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (unsigned int mask_index_y = 0; mask_index_y < mask_width; ++mask_index_y) {
                for (unsigned int mask_index_x = 0; mask_index_x < mask_width; ++mask_index_x) {
                    unsigned int mask_index = mask_index_y * mask_width + mask_index_x;
                    unsigned int input_index = (y + mask_index_y) * padded_width + (x + mask_index_x);
                    sum += input[input_index] * mask[mask_index];
                }
            }
            output[(y * width + x)] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    // Default values
    unsigned int width = 4096;
    unsigned int height = 4096;
    unsigned int num_threads = 8;

    // Check if user provided arguments
    if (argc > 1) {
        width = std::atoi(argv[1]);  // First argument: width
    }
    if (argc > 2) {
        height = std::atoi(argv[2]);  // Second argument: height
    }
    if (argc > 3) {
        num_threads = std::atoi(argv[3]);  // Third argument: number of threads
    }

    const unsigned int mask_width = 5;
    const unsigned int filter_radius = mask_width / 2;
    const unsigned int padded_width = width + filter_radius * 2;
    const unsigned int padded_height = height + filter_radius * 2;
    const unsigned int size = width * height;
    const unsigned int input_size_padded = padded_width * padded_height;

    std::cout<<"padded_width "<<padded_width<<"\n";

    std::vector<float> input_grid(size);
    std::vector<float> output_grid(size);
    std::vector<float> output_grid_single_threaded(size);
    std::vector<float> input_grid_padded(input_size_padded, 0);

    // Define a 5x5 convolution mask (convolution_filter_5x5)
    std::vector<float> convolution_filter_5x5 = {1.0f,  3.0f, 0.0f,  -2.0f, -0.0f,
                                                                   1.0f,  4.0f, 0.0f,  -8.0f, -4.0f,
                                                                   2.0f,  7.0f, 0.0f, -12.0f, -0.0f,
                                                                   2.0f,  3.0f, 1.5f,  -8.0f, -4.0f,
                                                                   0.0f,  1.0f, 0.0f,  -2.0f, -0.0f};

    // Generate random input
    std::mt19937 mersenne_engine{0};
    std::uniform_real_distribution<float> distribution{0, 256};
    auto rnd = std::bind(distribution, mersenne_engine);  // Use std::bind for random number generation
    std::generate(input_grid.begin(), input_grid.end(), rnd);

    // Copy input grid to padded input grid
    auto input_grid_row_begin = input_grid.begin();
    auto padded_input_grid_row_begin = input_grid_padded.begin() + filter_radius * padded_width + filter_radius;
    for (unsigned int i = 0; i < height; i++) {
        std::copy(input_grid_row_begin, input_grid_row_begin + width, padded_input_grid_row_begin);
        padded_input_grid_row_begin += padded_width;
        input_grid_row_begin += width;
    }

    // --- Multithreaded Convolution ---
    pthread_barrier_init(&barrier, nullptr, num_threads);  // Initialize barrier

    ThreadData thread_data[num_threads];
    pthread_t threads[num_threads];

    for (unsigned int i = 0; i < num_threads; ++i) {
        thread_data[i].input = input_grid_padded.data();
        thread_data[i].output = output_grid.data();
        thread_data[i].mask = convolution_filter_5x5.data();
        thread_data[i].width = width;
        thread_data[i].padded_width = padded_width;
        thread_data[i].mask_width = mask_width;
        thread_data[i].height = height;
        thread_data[i].thread_id = i;
        thread_data[i].columns_processed = 0;  // Initialize column counter
        pthread_create(&threads[i], nullptr, convolution_thread, &thread_data[i]);
    }

    // Join threads
    for (unsigned int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    pthread_barrier_destroy(&barrier);  // Destroy barrier

#ifdef DEBUG_CHECK
    // --- Single-threaded Convolution ---
    convolution_single_threaded(input_grid_padded.data(), output_grid_single_threaded.data(),
                                convolution_filter_5x5.data(), width, padded_width, mask_width, height);

    // --- Compare Results ---
    bool is_equal = std::equal(output_grid.begin(), output_grid.end(), output_grid_single_threaded.begin(),
                               [](float a, float b) { return std::fabs(a - b) < 1e-5; });

    if (is_equal) {
        std::cout << "The outputs are the same. Multithreaded version is correct." << std::endl;
    } else {
        std::cout << "The outputs are different. There may be an issue with the multithreaded version." << std::endl;
    }
#endif

    // Print number of columns processed by each thread
    for (unsigned int i = 0; i < num_threads; ++i) {
        std::cout << "Thread " << i << " processed " << thread_data[i].columns_processed.load() << " columns." << std::endl;
    }

    return 0;
}
