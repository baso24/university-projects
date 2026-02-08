#include <iostream>
#include <vector>
#include <chrono> // Per il timing ad alta precisione
#include <iomanip>
#include "mtx_reader.h"

// Numero di iterazioni per avere una media stabile
const int NUM_ITERATIONS = 1000;

// =================================================================================
// 1. Sequential Kernel: CSR
// =================================================================================
void Smv_CSR_Sequential(const CSRMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    
    // Puntatori diretti per evitare overhead di vector::operator[]
    const int* row_ptr = mat.row_ptr.data();
    const int* col_indices = mat.col_indices.data();
    const float* values = mat.values.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
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
    
    std::cout << "Sequential CSR Time: " << duration.count() / NUM_ITERATIONS << " ms" << std::endl;
}

// =================================================================================
// 2. Sequential Kernel: ELL (Row-Major)
// =================================================================================
void Smv_ELL_Sequential(const ELLMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    // Controllo sicurezza: La CPU lavora male con il layout GPU (Column-Major)
    if (mat.is_column_major) {
        std::cerr << "WARNING: Stai usando layout Column-Major su CPU. Le prestazioni saranno pessime (Cache Miss)." << std::endl;
    }

    int rows = mat.rows;
    int max_nnz = mat.max_nnz_per_row;
    bool is_col_major = mat.is_column_major;

    const float* values = mat.values.data();
    const int* col_indices = mat.col_indices.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int c = 0; c < max_nnz; c++) {
                int idx;
                // Calcolo indice manuale basato sul layout
                if (is_col_major) idx = r + c * rows; // Lento su CPU
                else              idx = r * max_nnz + c; // Veloce su CPU (Contiguo)

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

    std::cout << "Sequential ELL Time: " << duration.count() / NUM_ITERATIONS << " ms" << std::endl;
}

// =================================================================================
// 3. Sequential Kernel: Hybrid (ELL + COO)
// =================================================================================
void Smv_Hybrid_Sequential(const HybridMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
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

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        
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

    std::cout << "Sequential Hybrid Time: " << duration.count() / NUM_ITERATIONS << " ms" << std::endl;
}

void Smv_JDS_Sequential(const JDSMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows; // Numero righe totali
    const float* values = mat.values.data();
    const int* col_indices = mat.col_indices.data();
    const int* jd_ptr = mat.jd_ptr.data();
    const int* row_lengths = mat.row_lengths.data();
    const int* row_perm = mat.row_perm.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Reset vettore Y (necessario se accumuliamo, ma qui sovrascriviamo riga per riga)
        // Nota: JDS classico calcola per diagonali, ma per semplicità ed evitare race condition 
        // in parallelo, spesso si itera sulle righe JDS.
        // Qui usiamo l'approccio "Row-wise on Permuted Matrix" che è cache-friendly per x
        
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            int len = row_lengths[i];
            
            // Itera lungo la riga 'i' della matrice JDS.
            // In memoria JDS, l'elemento k-esimo della riga i si trova saltando usando jd_ptr
            for (int k = 0; k < len; k++) {
                // Calcolo indice: Inizio della diagonale k + offset riga i
                int idx = jd_ptr[k] + i; 
                sum += values[idx] * x_ptr[col_indices[idx]];
            }
            
            // Scatter: Scrittura nella posizione originale
            y_ptr[row_perm[i]] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Sequential JDS Time: " << duration.count() / NUM_ITERATIONS << " ms" << std::endl;
}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./seq_benchmark <matrix.mtx> <hybrid_cutoff>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int cutoff = std::stoi(argv[2]);
    int M, N, nnz;

    // 1. Lettura
    std::cout << "Reading " << filename << "..." << std::endl;
    auto raw_elements = read_mtx_file(filename, M, N, nnz);

    // 2. Setup Vettori
    std::vector<float> x(N, 1.0f);
    std::vector<float> y_out(M, 0.0f);

    // --- TEST CSR ---
    CSRMatrix mat_csr = convert_to_csr(raw_elements, M, N);
    
    
    //ELLMatrix mat_ell = convert_to_ell(raw_elements, M, mat_csr.row_ptr, false);
    HybridMatrix mat_hyb = convert_to_hybrid(raw_elements, M, N, cutoff, false);
    JDSMatrix mat_jds = convert_to_jds(raw_elements, M, N, mat_csr.row_ptr);

    // --- TEST ELL (Row-Major per CPU) ---
    // NOTA: Passiamo 'false' come ultimo argomento per avere Row-Major
    
    Smv_CSR_Sequential(mat_csr, x, y_out);
    /*
    if (mat_ell.rows > 0) {
        
        Smv_ELL_Sequential(mat_ell, x, y_out);
    } else {
        std::cout << "ELL Skipped (Too large)." << std::endl;
    }
    */
    // --- TEST HYBRID (Row-Major per CPU) ---
    // Anche qui 'false' per Row-Major
    
    print_matrix_info(mat_csr, ELLMatrix(), mat_hyb, mat_jds);
   
    //std::cout << "Hybrid Stats -> ELL Part: " << (nnz - mat_hyb.coo_nnz) 
    //          << ", COO Part: " << mat_hyb.coo_nnz << " (" << coo_ratio << "%)" << std::endl;

    Smv_Hybrid_Sequential(mat_hyb, x, y_out);
    
    Smv_JDS_Sequential(mat_jds, x, y_out);
    return 0;
}