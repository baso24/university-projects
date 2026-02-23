#include <iostream>
#include <vector>
#include <chrono> // Per il timing ad alta precisione
#include <iomanip>
#include "mtx_reader.h"


float Smv_CSR_Sequential(const CSRMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    
    const int* row_ptr = mat.row_ptr.data();
    const int* col_indices = mat.col_indices.data();
    const float* values = mat.values.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 1000; iter++) {
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            int start_idx = row_ptr[i];
            int end_idx = row_ptr[i+1];
            
            for (int j = start_idx; j < end_idx; j++) {
                sum += values[j] * x_ptr[col_indices[j]];
            }
            y_ptr[i] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    return duration.count() / 1000.0f;
}

float Smv_ELL_Sequential(const ELLMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {

    int rows = mat.rows;
    int max_nnz = mat.max_nnz_per_row;
    bool is_col_major = mat.is_column_major;

    const float* values = mat.values.data();
    const int* col_indices = mat.col_indices.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 1000; iter++) {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int c = 0; c < max_nnz; c++) {
                int idx;
                if (is_col_major) 
                    idx = r + c * rows;
                else              
                    idx = r * max_nnz + c;

                int col = col_indices[idx];
                
                // -1 indica padding
                if (col != -1) {
                    sum += values[idx] * x_ptr[col];
                }
            }
            y_ptr[r] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    return duration.count() / 1000.0f;
}

float Smv_Hybrid_Sequential(const HybridMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    int max_nnz = mat.ell_part.max_nnz_per_row;
    
    const float* ell_val = mat.ell_part.values.data();
    const int* ell_col = mat.ell_part.col_indices.data();
    
    const float* coo_val = mat.coo_values.data();
    const int* coo_row = mat.coo_row_indices.data();
    const int* coo_col = mat.coo_col_indices.data();
    int coo_nnz = mat.coo_nnz;

    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 1000; iter++) {
        
        // FASE 1: ELL (Base)
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int c = 0; c < max_nnz; c++) {
                int idx = r * max_nnz + c; // Assumiamo Row-Major qui per performance
                int col = ell_col[idx];
                if (col != -1) {
                    sum += ell_val[idx] * x_ptr[col];
                }
            }
            y_ptr[r] = sum;
        }

        // FASE 2: COO (Overflow)
        // In sequenziale non servono atomiche, basta sommare
        for (int i = 0; i < coo_nnz; i++) {
            int r = coo_row[i];
            int c = coo_col[i];
            y_ptr[r] += coo_val[i] * x_ptr[c];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    return duration.count() / 1000.0f;
}

float Smv_JDS_Sequential(const JDSMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows; // Numero righe totali
    const float* values = mat.values.data();
    const int* col_indices = mat.col_indices.data();
    const int* jd_ptr = mat.jd_ptr.data();
    const int* row_lengths = mat.row_lengths.data();
    const int* row_perm = mat.row_perm.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 1000; iter++) {        
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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / 1000.0f;
}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./seq_benchmark <matrix.mtx> <hybrid_cutoff>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int cutoff = std::stoi(argv[2]);
    int M, N, nnz;

    std::cout << "Reading " << filename << "..." << std::endl;
    auto raw_elements = read_mtx_file(filename, M, N, nnz);

    std::vector<float> x(N, 1.0f);
    std::vector<float> y_out(M, 0.0f);

    CSRMatrix mat_csr = convert_to_csr(raw_elements, M, N);
    ELLMatrix mat_ell = convert_to_ell(raw_elements, M, mat_csr.row_ptr, false);  
    HybridMatrix mat_hyb = convert_to_hybrid(raw_elements, M, N, cutoff, false);
    JDSMatrix mat_jds = convert_to_jds(raw_elements, M, N, mat_csr.row_ptr);

    print_matrix_info(mat_csr, ELLMatrix(), mat_hyb, mat_jds);

    Smv_CSR_Sequential(mat_csr, x, y_out);
    Smv_ELL_Sequential(mat_hyb.ell_part, x, y_out);
    Smv_Hybrid_Sequential(mat_hyb, x, y_out);
    Smv_JDS_Sequential(mat_jds, x, y_out);
    return 0;
}