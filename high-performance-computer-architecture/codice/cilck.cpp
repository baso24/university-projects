#include <iostream>
#include <vector>
#include <cilk/cilk.h>
#include <chrono>
#include <atomic>
#include "mtx_reader.h"

// Numero di iterazioni per il benchmark
const int NUM_ITERATIONS = 1000;

// Helper per addizione atomica su float (Cilk non ha #pragma omp atomic)
// Usa un loop Compare-And-Swap (CAS) per garantire la correttezza
inline void atomic_add_float(float* addr, float value) {
    float current, target;
    // Caricamento atomico del valore corrente
    // Nota: usiamo __atomic builtins che sono standard su Clang/OpenCilk e GCC
    __atomic_load(addr, &current, __ATOMIC_RELAXED);
    do {
        target = current + value;
        // Tenta di scambiare il valore. Se fallisce (qualcun altro ha scritto),
        // aggiorna 'current' con il nuovo valore e riprova.
    } while (!__atomic_compare_exchange(addr, &current, &target, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}

// =================================================================================
// 1. Cilk Kernel: CSR
// =================================================================================
void Smv_CSR_Cilk(const CSRMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    
    const int* __restrict__ row_ptr = mat.row_ptr.data();
    const int* __restrict__ col_indices = mat.col_indices.data();
    const float* __restrict__ values = mat.values.data();
    const float* __restrict__ x_ptr = x.data();
    float* __restrict__ y_ptr = y_out.data();

    // Warm-up
    cilk_for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
            sum += values[j] * x_ptr[col_indices[j]];
        }
        y_ptr[i] = sum;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // cilk_for parallelizza automaticamente il loop esterno
        cilk_for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            int start_idx = row_ptr[i];
            int end_idx = row_ptr[i+1];
            
            // Loop interno sequenziale (vettorializzabile dal compilatore)
            for (int j = start_idx; j < end_idx; j++) {
                sum += values[j] * x_ptr[col_indices[j]];
            }
            y_ptr[i] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "Cilk CSR Time: " << duration.count() / NUM_ITERATIONS << " ms" << std::endl;
}

// =================================================================================
// 2. Cilk Kernel: ELL (Row-Major per CPU)
// =================================================================================
void Smv_ELL_Cilk(const ELLMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    if (mat.is_column_major) {
        std::cerr << "Errore: Cilk richiede layout Row-Major per efficienza cache!" << std::endl;
        return;
    }

    int rows = mat.rows;
    int max_nnz = mat.max_nnz_per_row;

    const float* __restrict__ values = mat.values.data();
    const int* __restrict__ col_indices = mat.col_indices.data();
    const float* __restrict__ x_ptr = x.data();
    float* __restrict__ y_ptr = y_out.data();

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        cilk_for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            // Accesso Row-Major contiguo
            for (int c = 0; c < max_nnz; c++) {
                int idx = r * max_nnz + c;
                int col = col_indices[idx];
                
                if (col != -1) {
                    sum += values[idx] * x_ptr[col];
                }
            }
            y_ptr[r] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Cilk ELL Time: " << duration.count() / NUM_ITERATIONS << " ms" << std::endl;
}

// =================================================================================
// 3. Cilk Kernel: Hybrid (ELL + COO)
// =================================================================================
void Smv_Hybrid_Cilk(const HybridMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    int max_nnz = mat.ell_part.max_nnz_per_row;
    
    const float* __restrict__ ell_val = mat.ell_part.values.data();
    const int* __restrict__ ell_col = mat.ell_part.col_indices.data();
    
    const float* __restrict__ coo_val = mat.coo_values.data();
    const int* __restrict__ coo_row = mat.coo_row_indices.data();
    const int* __restrict__ coo_col = mat.coo_col_indices.data();
    int coo_nnz = mat.coo_nnz;

    const float* __restrict__ x_ptr = x.data();
    float* __restrict__ y_ptr = y_out.data();

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        
        // FASE 1: ELL (Inizializza y_ptr in parallelo)
        // Nessuna race condition qui, ogni iterazione scrive su riga distinta
        cilk_for (int r = 0; r < rows; r++) {
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

        // FASE 2: COO (Overflow - Additivo)
        // Race conditions possibili! Usiamo atomics.
        if (coo_nnz > 0) {
            cilk_for (int i = 0; i < coo_nnz; i++) {
                int r = coo_row[i];
                int c = coo_col[i];
                float val = coo_val[i] * x_ptr[c];
                
                // Aggiornamento atomico custom
                atomic_add_float(&y_ptr[r], val);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Cilk Hybrid Time: " << duration.count() / NUM_ITERATIONS << " ms" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./cilk_benchmark <matrix.mtx> <hybrid_cutoff>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int cutoff = std::stoi(argv[2]);
    int M, N, nnz;

    // Impostare i worker Cilk da riga di comando o variabile ambiente
    // CILK_NWORKERS=4 ./cilk_benchmark ...

    // 1. Lettura Dati
    std::cout << "Reading " << filename << "..." << std::endl;
    auto raw_elements = read_mtx_file(filename, M, N, nnz);

    // 2. Setup Vettori
    std::vector<float> x(N, 1.0f);
    std::vector<float> y_out(M, 0.0f);

    // ------------------------------------------
    // TEST CSR
    // ------------------------------------------
    CSRMatrix mat_csr = convert_to_csr(raw_elements, M, N);
    
    size_t mem_csr = (mat_csr.values.size() * sizeof(float)) + 
                     (mat_csr.col_indices.size() * sizeof(int)) + 
                     (mat_csr.row_ptr.size() * sizeof(int));
    std::cout << "CSR Memory: " << mem_csr / 1024.0 / 1024.0 << " MB" << std::endl;
    
    Smv_CSR_Cilk(mat_csr, x, y_out);


    // ------------------------------------------
    // TEST ELL (CPU Version - Row Major)
    // ------------------------------------------
    // Importante: false per layout Row-Major
    ELLMatrix mat_ell = convert_to_ell(raw_elements, M, mat_csr.row_ptr, false); 
    
    if (mat_ell.rows > 0) {
        print_matrix_info(mat_csr, mat_ell);
        Smv_ELL_Cilk(mat_ell, x, y_out);
    } else {
        std::cout << "ELL Skipped (Too large)." << std::endl;
    }

    // ------------------------------------------
    // TEST HYBRID (CPU Version - Row Major)
    // ------------------------------------------
    HybridMatrix mat_hyb = convert_to_hybrid(raw_elements, M, N, cutoff, false);
    
    float coo_ratio = (float)mat_hyb.coo_nnz / (float)nnz * 100.0f;
    std::cout << "Hybrid Stats -> ELL Part: " << (nnz - mat_hyb.coo_nnz) 
              << " nnz, COO Part: " << mat_hyb.coo_nnz << " nnz (" << coo_ratio << "%)" << std::endl;

    Smv_Hybrid_Cilk(mat_hyb, x, y_out);

    return 0;
}