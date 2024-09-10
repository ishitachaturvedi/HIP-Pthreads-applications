#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>  // For atoi and atof

__global__ void sparse_matmul_hip_kernel(int* row_ptr_A, int* col_ind_A, double* val_A, 
                                         int* row_ptr_B, int* col_ind_B, double* val_B, 
                                         double* result, int A_rows, int B_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows) {
        for (int k = row_ptr_A[row]; k < row_ptr_A[row + 1]; k++) {
            int a_col = col_ind_A[k];
            double a_val = val_A[k];

            for (int j = row_ptr_B[a_col]; j < row_ptr_B[a_col + 1]; j++) {
                int b_col = col_ind_B[j];
                double b_val = val_B[j];

                atomicAdd(&result[row * B_cols + b_col], a_val * b_val);
            }
        }
    }
}

void sparse_matmul_hip(int* row_ptr_A, [[maybe_unused]] int* col_ind_A, 
                       [[maybe_unused]] double* val_A, int A_rows, 
                       int* row_ptr_B, [[maybe_unused]] int* col_ind_B, 
                       [[maybe_unused]] double* val_B, int B_cols, 
                       [[maybe_unused]] double* result) {
    int *d_row_ptr_A, *d_col_ind_A, *d_row_ptr_B, *d_col_ind_B;
    double *d_val_A, *d_val_B, *d_result;

    // Allocate pinned memory on the host
    if (hipHostMalloc(&d_row_ptr_A, (A_rows + 1) * sizeof(int)) != hipSuccess) {
        std::cerr << "Error allocating memory for d_row_ptr_A\n";
        return;
    }
    if (hipHostMalloc(&d_col_ind_A, row_ptr_A[A_rows] * sizeof(int)) != hipSuccess) {
        std::cerr << "Error allocating memory for d_col_ind_A\n";
        return;
    }
    if (hipHostMalloc(&d_val_A, row_ptr_A[A_rows] * sizeof(double)) != hipSuccess) {
        std::cerr << "Error allocating memory for d_val_A\n";
        return;
    }

    if (hipHostMalloc(&d_row_ptr_B, (B_cols + 1) * sizeof(int)) != hipSuccess) {
        std::cerr << "Error allocating memory for d_row_ptr_B\n";
        return;
    }
    if (hipHostMalloc(&d_col_ind_B, row_ptr_B[B_cols] * sizeof(int)) != hipSuccess) {
        std::cerr << "Error allocating memory for d_col_ind_B\n";
        return;
    }
    if (hipHostMalloc(&d_val_B, row_ptr_B[B_cols] * sizeof(double)) != hipSuccess) {
        std::cerr << "Error allocating memory for d_val_B\n";
        return;
    }

    if (hipHostMalloc(&d_result, A_rows * B_cols * sizeof(double)) != hipSuccess) {
        std::cerr << "Error allocating memory for d_result\n";
        return;
    }

    if (hipMemset(d_result, 0, A_rows * B_cols * sizeof(double)) != hipSuccess) {
        std::cerr << "Error setting memory for d_result\n";
        return;
    }

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (A_rows + threads_per_block - 1) / threads_per_block;
    sparse_matmul_hip_kernel<<<blocks, threads_per_block>>>(d_row_ptr_A, d_col_ind_A, d_val_A, 
                                                            d_row_ptr_B, d_col_ind_B, d_val_B, 
                                                            d_result, A_rows, B_cols);

    if (hipDeviceSynchronize() != hipSuccess) {
        std::cerr << "Error synchronizing device\n";
        return;
    }

    // Copy result back to host
    std::vector<double> result_host(A_rows * B_cols);
    if (hipMemcpy(result_host.data(), d_result, A_rows * B_cols * sizeof(double), hipMemcpyDeviceToHost) != hipSuccess) {
        std::cerr << "Error copying result back to host\n";
        return;
    }

    // Print result matrix
    // std::cout << "Result Matrix:\n";
    // for (int i = 0; i < A_rows; i++) {
    //     for (int j = 0; j < B_cols; j++) {
    //         std::cout << result_host[i * B_cols + j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // Free device memory
    if (hipHostFree(d_row_ptr_A) != hipSuccess) {
        std::cerr << "Error freeing memory for d_row_ptr_A\n";
    }
    if (hipHostFree(d_col_ind_A) != hipSuccess) {
        std::cerr << "Error freeing memory for d_col_ind_A\n";
    }
    if (hipHostFree(d_val_A) != hipSuccess) {
        std::cerr << "Error freeing memory for d_val_A\n";
    }
    if (hipHostFree(d_row_ptr_B) != hipSuccess) {
        std::cerr << "Error freeing memory for d_row_ptr_B\n";
    }
    if (hipHostFree(d_col_ind_B) != hipSuccess) {
        std::cerr << "Error freeing memory for d_col_ind_B\n";
    }
    if (hipHostFree(d_val_B) != hipSuccess) {
        std::cerr << "Error freeing memory for d_val_B\n";
    }
    if (hipHostFree(d_result) != hipSuccess) {
        std::cerr << "Error freeing memory for d_result\n";
    }
}

void generate_sparse_matrix(int rows, int cols, double sparsity, unsigned seed, 
                            std::vector<int>& row_ptr, std::vector<int>& col_ind, std::vector<double>& val) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis_val(0.0, 10.0);
    std::uniform_real_distribution<> dis_spar(0.0, 1.0);

    row_ptr.resize(rows + 1);
    int nnz = 0;

    for (int i = 0; i < rows; i++) {
        row_ptr[i] = nnz;
        for (int j = 0; j < cols; j++) {
            if (dis_spar(gen) >= sparsity) {
                col_ind.push_back(j);
                val.push_back(dis_val(gen));
                nnz++;
            }
        }
    }
    row_ptr[rows] = nnz;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <A_rows> <B_cols> <sparsity_A> <sparsity_B>\n";
        return 1;
    }

    int A_rows = atoi(argv[1]);
    int B_cols = atoi(argv[2]);
    double sparsity_A = atof(argv[3]);
    double sparsity_B = atof(argv[4]);
    unsigned seed = 2;

    std::vector<int> row_ptr_A, col_ind_A, row_ptr_B, col_ind_B;
    std::vector<double> val_A, val_B;

    // Generate random sparse matrices A and B with a specific seed
    generate_sparse_matrix(A_rows, B_cols, sparsity_A, seed, row_ptr_A, col_ind_A, val_A);
    generate_sparse_matrix(B_cols, A_rows, sparsity_B, seed, row_ptr_B, col_ind_B, val_B);

    std::vector<double> result(A_rows * B_cols, 0.0);

    // Perform sparse matrix multiplication
    sparse_matmul_hip(row_ptr_A.data(), col_ind_A.data(), val_A.data(), A_rows, 
                      row_ptr_B.data(), col_ind_B.data(), val_B.data(), B_cols, 
                      result.data());

    return 0;
}
