#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>

#define CHUNK_SIZE 10

typedef struct {
    int* row_ptr_A;
    int* col_ind_A;
    double* val_A;
    int* row_ptr_B;
    int* col_ind_B;
    double* val_B;
    double* result;
    int A_rows, B_cols, num_threads;
    int current_row;
    pthread_mutex_t row_lock;
} thread_data;

void* multiply(void* arg) {
    thread_data* data = (thread_data*)arg;
    int start_row;

    while (1) {
        pthread_mutex_lock(&data->row_lock);
        start_row = data->current_row;
        data->current_row += CHUNK_SIZE;
        pthread_mutex_unlock(&data->row_lock);

        if (start_row >= data->A_rows) {
            break;
        }

        int end_row = start_row + CHUNK_SIZE;
        if (end_row > data->A_rows) {
            end_row = data->A_rows;
        }

        for (int i = start_row; i < end_row; i++) {
            for (int k = data->row_ptr_A[i]; k < data->row_ptr_A[i + 1]; k++) {
                int a_col = data->col_ind_A[k];
                double a_val = data->val_A[k];

                for (int j = data->row_ptr_B[a_col]; j < data->row_ptr_B[a_col + 1]; j++) {
                    int b_col = data->col_ind_B[j];
                    double b_val = data->val_B[j];
                    data->result[i * data->B_cols + b_col] += a_val * b_val;
                }
            }
        }
    }

    pthread_exit(NULL);
}

void sparse_matmul_pthreads(int* row_ptr_A, int* col_ind_A, double* val_A, int A_rows, 
                            int* row_ptr_B, int* col_ind_B, double* val_B, int B_cols, 
                            int num_threads, double* result) {
    pthread_t threads[num_threads];
    thread_data t_data;
    
    t_data.row_ptr_A = row_ptr_A;
    t_data.col_ind_A = col_ind_A;
    t_data.val_A = val_A;
    t_data.A_rows = A_rows;
    t_data.row_ptr_B = row_ptr_B;
    t_data.col_ind_B = col_ind_B;
    t_data.val_B = val_B;
    t_data.B_cols = B_cols;
    t_data.result = result;
    t_data.num_threads = num_threads;
    t_data.current_row = 0;
    pthread_mutex_init(&t_data.row_lock, NULL);

    for (size_t i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, multiply, &t_data);
    }

    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&t_data.row_lock);
}

void generate_sparse_matrix(int rows, int cols, double sparsity, 
                            int** row_ptr, int** col_ind, double** val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_val(0.0, 10.0);
    std::uniform_real_distribution<> dis_spar(0.0, 1.0);

    *row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    std::vector<int> col_indices;
    std::vector<double> values;
    int nnz = 0;

    for (int i = 0; i < rows; i++) {
        (*row_ptr)[i] = nnz;
        for (int j = 0; j < cols; j++) {
            if (dis_spar(gen) >= sparsity) {
                col_indices.push_back(j);
                values.push_back(dis_val(gen));
                nnz++;
            }
        }
    }
    (*row_ptr)[rows] = nnz;

    *col_ind = (int*)malloc(nnz * sizeof(int));
    *val = (double*)malloc(nnz * sizeof(double));

    for (int i = 0; i < nnz; i++) {
        (*col_ind)[i] = col_indices[i];
        (*val)[i] = values[i];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        printf("Usage: %s <A_rows> <B_cols> <sparsity_A> <sparsity_B> <num_threads>\n", argv[0]);
        return 1;
    }

    int A_rows = atoi(argv[1]);
    int B_cols = atoi(argv[2]);
    double sparsity_A = atof(argv[3]);
    double sparsity_B = atof(argv[4]);
    int num_threads = atoi(argv[5]);

    int *row_ptr_A, *col_ind_A, *row_ptr_B, *col_ind_B;
    double *val_A, *val_B;

    // Generate sparse matrices A and B
    generate_sparse_matrix(A_rows, B_cols, sparsity_A, &row_ptr_A, &col_ind_A, &val_A);
    generate_sparse_matrix(B_cols, A_rows, sparsity_B, &row_ptr_B, &col_ind_B, &val_B);

    double* result = (double*)calloc(A_rows * B_cols, sizeof(double));

    // Perform matrix multiplication using pthreads
    sparse_matmul_pthreads(row_ptr_A, col_ind_A, val_A, A_rows, row_ptr_B, col_ind_B, val_B, B_cols, num_threads, result);

    // Print the result matrix
    // printf("Result Matrix:\n");
    // for (int i = 0; i < A_rows; i++) {
    //     for (int j = 0; j < B_cols; j++) {
    //         printf("%lf ", result[i * B_cols + j]);
    //     }
    //     printf("\n");
    // }

    free(row_ptr_A);
    free(col_ind_A);
    free(val_A);
    free(row_ptr_B);
    free(col_ind_B);
    free(val_B);
    free(result);

    return 0;
}
