#include <iostream>
#include <vector>
#include <pthread.h>
#include <cstdlib>
#include <climits>

// Struct to hold thread data
struct ThreadData {
    unsigned int* adjacency_matrix;
    unsigned int* next_matrix;
    unsigned int nodes;
    unsigned int k;
    unsigned int start_row;
    unsigned int end_row;
};

// Floyd-Warshall kernel function for each thread
void* floyd_warshall_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    unsigned int* adjacency_matrix = data->adjacency_matrix;
    unsigned int* next_matrix = data->next_matrix;
    unsigned int nodes = data->nodes;
    unsigned int k = data->k;
    unsigned int start_row = data->start_row;
    unsigned int end_row = data->end_row;

    for(unsigned int x = start_row; x < end_row; x++) {
        const unsigned int row_x = x * nodes;
        for(unsigned int y = 0; y < nodes; y++) {
            const unsigned int d_x_y = adjacency_matrix[row_x + y];
            const unsigned int d_x_k = adjacency_matrix[row_x + k];
            const unsigned int d_k_y = adjacency_matrix[k * nodes + y];
            const unsigned int d_x_k_y = d_x_k + d_k_y;

            if(d_x_k_y < d_x_y) {
                adjacency_matrix[row_x + y] = d_x_k_y;
                next_matrix[row_x + y] = k;
            }
        }
    }
    return nullptr;
}

// Reference CPU implementation for validation
void floyd_warshall_reference(unsigned int* adjacency_matrix,
                              unsigned int* next_matrix,
                              const unsigned int nodes) {
    for(unsigned int k = 0; k < nodes; k++) {
        for(unsigned int x = 0; x < nodes; x++) {
            const unsigned int row_x = x * nodes;
            for(unsigned int y = 0; y < nodes; y++) {
                const unsigned int d_x_y = adjacency_matrix[row_x + y];
                const unsigned int d_x_k = adjacency_matrix[row_x + k];
                const unsigned int d_k_y = adjacency_matrix[k * nodes + y];
                const unsigned int d_x_k_y = d_x_k + d_k_y;

                if(d_x_k_y < d_x_y) {
                    adjacency_matrix[row_x + y] = d_x_k_y;
                    next_matrix[row_x + y] = k;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if(argc < 4) {
        std::cout << "Usage: " << argv[0] << " <nodes> <iterations> <num_threads>" << std::endl;
        return -1;
    }

    const unsigned int nodes = std::stoi(argv[1]);
    const unsigned int iterations = std::stoi(argv[2]);
    const unsigned int num_threads = std::stoi(argv[3]);

    if(num_threads == 0) {
        std::cout << "Number of threads must be greater than 0." << std::endl;
        return -1;
    }

    const unsigned int size = nodes * nodes;

    std::vector<unsigned int> adjacency_matrix(size);
    std::vector<unsigned int> next_matrix(size);

    for(unsigned int x = 0; x < nodes; x++) {
        for(unsigned int y = 0; y < nodes; y++) {
            adjacency_matrix[x * nodes + y] = (x == y) ? 0 : rand() % 100 + 1;
            next_matrix[x * nodes + y] = y;
        }
    }

    std::vector<unsigned int> expected_adjacency_matrix(adjacency_matrix);
    std::vector<unsigned int> expected_next_matrix(next_matrix);

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    for(unsigned int i = 0; i < iterations; ++i) {
        for(unsigned int k = 0; k < nodes; ++k) {
            unsigned int rows_per_thread = nodes / num_threads;

            for(unsigned int t = 0; t < num_threads; ++t) {
                unsigned int start_row = t * rows_per_thread;
                unsigned int end_row = (t == num_threads - 1) ? nodes : start_row + rows_per_thread;

                thread_data[t] = {adjacency_matrix.data(), next_matrix.data(), nodes, k, start_row, end_row};
                pthread_create(&threads[t], nullptr, floyd_warshall_thread, &thread_data[t]);
            }

            for(unsigned int t = 0; t < num_threads; ++t) {
                pthread_join(threads[t], nullptr);
            }
        }
    }

    floyd_warshall_reference(expected_adjacency_matrix.data(), expected_next_matrix.data(), nodes);

    unsigned int errors = 0;
    for(unsigned int i = 0; i < size; ++i) {
        errors += (adjacency_matrix[i] != expected_adjacency_matrix[i]);
    }

    if(errors) {
        std::cout << "Validation failed with " << errors << " errors." << std::endl;
        return -1;
    } else {
        std::cout << "Validation passed." << std::endl;
    }

    return 0;
}
