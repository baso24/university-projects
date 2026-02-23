#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include "mtx_reader.h"

float Smv_CSR_OpenMP(const CSRMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    
    // puntatori
    const int* row_ptr = mat.row_ptr.data();
    const int* col_indices = mat.col_indices.data();
    const float* values = mat.values.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
            sum += values[j] * x_ptr[col_indices[j]];
        }
        y_ptr[i] = sum;
    }

    double start = omp_get_wtime();

    for (int iter = 0; iter < 1000; iter++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            int start_idx = row_ptr[i];
            int end_idx = row_ptr[i+1];
            
            #pragma omp simd reduction(+:sum)
            for (int j = start_idx; j < end_idx; j++) {
                sum += values[j] * x_ptr[col_indices[j]];
            }
            y_ptr[i] = sum;
        }
    }

    double end = omp_get_wtime();
    double avg_time_ms = ((end - start) / 1000) * 1000.0;
    
    return avg_time_ms;
}

float Smv_ELL_OpenMP(const ELLMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    if (mat.is_column_major) {
        std::cerr << "Errore: OpenMP richiede layout Row-Major per ELL!" << std::endl;
        return 0;
    }

    int rows = mat.rows;
    int max_nnz = mat.max_nnz_per_row;

    const float* values = mat.values.data();
    const int* col_indices = mat.col_indices.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    double start = omp_get_wtime();

    for (int iter = 0; iter < 1000; iter++) {
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            // Accesso Row-Major: r * max_nnz + c
            for (int c = 0; c < max_nnz; c++) {
                int idx = r * max_nnz + c;
                int col = col_indices[idx];
                
                // Controllo padding (-1)
                if (col != -1) {
                    sum += values[idx] * x_ptr[col];
                }
            }
            y_ptr[r] = sum; // In ELL puro sovrascriviamo
        }
    }

    double end = omp_get_wtime();
    double avg_time_ms = ((end - start) / 1000) * 1000.0;

    std::cout << "OpenMP ELL Time: " << avg_time_ms << " ms" << std::endl;
    return avg_time_ms;
}

float Smv_Hybrid_OpenMP(const HybridMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    int max_nnz = mat.ell_part.max_nnz_per_row;
    
    // puntatori ELL
    const float* ell_val = mat.ell_part.values.data();
    const int* ell_col = mat.ell_part.col_indices.data();
    
    // puntatori COO
    const float* coo_val = mat.coo_values.data();
    const int* coo_row = mat.coo_row_indices.data();
    const int* coo_col = mat.coo_col_indices.data();
    int coo_nnz = mat.coo_nnz;

    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    double start = omp_get_wtime();

    for (int iter = 0; iter < 1000; iter++) {
        // ell
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int c = 0; c < max_nnz; c++) {
                int idx = r * max_nnz + c; 
                int col = ell_col[idx];
                if (col != -1) {
                    sum += ell_val[idx] * x_ptr[col];
                }
            }
            y_ptr[r] = sum;
        }

        // coo
        if (coo_nnz > 0) {
            #pragma omp parallel for
            for (int i = 0; i < coo_nnz; i++) {
                int r = coo_row[i];
                int c = coo_col[i];
                float val = coo_val[i];
                
                // Aggiornamento atomico per evitare race conditions
                #pragma omp atomic update
                y_ptr[r] += val * x_ptr[c];
            }
        }
    }

    double end = omp_get_wtime();
    double avg_time_ms = ((end - start) / 1000) * 1000.0;

    std::cout << "OpenMP Hybrid Time: " << avg_time_ms << " ms" << std::endl;
    return avg_time_ms;
}

float Smv_JDS_OpenMP(const JDSMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    const float* values = mat.values.data();
    const int* col_indices = mat.col_indices.data();
    const int* jd_ptr = mat.jd_ptr.data();
    const int* row_lengths = mat.row_lengths.data();
    const int* row_perm = mat.row_perm.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    double start = omp_get_wtime();

    for (int iter = 0; iter < 1000; iter++) {
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            int len = row_lengths[i];
            
            for (int k = 0; k < len; k++) {
                int idx = jd_ptr[k] + i;
                sum += values[idx] * x_ptr[col_indices[idx]];
            }
            y_ptr[row_perm[i]] = sum;
        }
    }

    double end = omp_get_wtime();
    double avg_time_ms = ((end - start) / 1000) * 1000.0;
    std::cout << "OpenMP JDS Time: " << avg_time_ms << " ms" << std::endl;
    return avg_time_ms;
}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./omp_benchmark <matrix.mtx> <ell_cutoff>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int cutoff = std::stoi(argv[2]);
    int M, N, nnz;
    std::cout << "OpenMP Threads Max: " << omp_get_max_threads() << std::endl;

    std::cout << "Reading " << filename << "..." << std::endl;
    auto raw_elements = read_mtx_file(filename, M, N, nnz);

    std::vector<float> x(N, 1.0f);
    std::vector<float> y_out(M, 0.0f);

    CSRMatrix mat_csr = convert_to_csr(raw_elements, M, N);
    ELLMatrix mat_ell = convert_to_ell(raw_elements, M, mat_csr.row_ptr, true); 
    HybridMatrix mat_hyb = convert_to_hybrid(raw_elements, M, N, cutoff, true);
    JDSMatrix mat_jds = convert_to_jds(raw_elements, M, N, mat_csr.row_ptr);
    print_matrix_info(mat_csr, ELLMatrix() , mat_hyb, mat_jds);
    Smv_CSR_OpenMP(mat_csr, x, y_out);
    Smv_Hybrid_OpenMP(mat_hyb, x, y_out);
    Smv_JDS_OpenMP(mat_jds, x, y_out);
    return 0;
}