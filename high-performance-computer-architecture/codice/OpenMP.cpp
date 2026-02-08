#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include "mtx_reader.h"

// Numero di iterazioni per il benchmark (per mediare il tempo)
const int NUM_ITERATIONS = 1000;

// =================================================================================
// 1. OpenMP Kernel: CSR
// =================================================================================
void Smv_CSR_OpenMP(const CSRMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    
    // Puntatori grezzi per velocità
    const int* __restrict__ row_ptr = mat.row_ptr.data();
    const int* __restrict__ col_indices = mat.col_indices.data();
    const float* __restrict__ values = mat.values.data();
    const float* __restrict__ x_ptr = x.data();
    float* __restrict__ y_ptr = y_out.data();

    // Warm-up
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
            sum += values[j] * x_ptr[col_indices[j]];
        }
        y_ptr[i] = sum;
    }

    double start = omp_get_wtime();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            int start_idx = row_ptr[i];
            int end_idx = row_ptr[i+1];
            
            // Loop interno vettorializzabile dal compilatore
            #pragma omp simd reduction(+:sum)
            for (int j = start_idx; j < end_idx; j++) {
                sum += values[j] * x_ptr[col_indices[j]];
            }
            y_ptr[i] = sum;
        }
    }

    double end = omp_get_wtime();
    double avg_time_ms = ((end - start) / NUM_ITERATIONS) * 1000.0;
    
    std::cout << "OpenMP CSR Time: " << avg_time_ms << " ms" << std::endl;
}

// =================================================================================
// 2. OpenMP Kernel: ELL (Row-Major per CPU)
// =================================================================================
void Smv_ELL_OpenMP(const ELLMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    if (mat.is_column_major) {
        std::cerr << "Errore: OpenMP richiede layout Row-Major per ELL!" << std::endl;
        return;
    }

    int rows = mat.rows;
    int max_nnz = mat.max_nnz_per_row;

    const float* __restrict__ values = mat.values.data();
    const int* __restrict__ col_indices = mat.col_indices.data();
    const float* __restrict__ x_ptr = x.data();
    float* __restrict__ y_ptr = y_out.data();

    double start = omp_get_wtime();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
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
    double avg_time_ms = ((end - start) / NUM_ITERATIONS) * 1000.0;

    std::cout << "OpenMP ELL Time: " << avg_time_ms << " ms" << std::endl;
}

// =================================================================================
// 3. OpenMP Kernel: Hybrid (ELL + COO)
// =================================================================================
void Smv_Hybrid_OpenMP(const HybridMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    int max_nnz = mat.ell_part.max_nnz_per_row;
    
    // Puntatori ELL
    const float* __restrict__ ell_val = mat.ell_part.values.data();
    const int* __restrict__ ell_col = mat.ell_part.col_indices.data();
    
    // Puntatori COO
    const float* __restrict__ coo_val = mat.coo_values.data();
    const int* __restrict__ coo_row = mat.coo_row_indices.data();
    const int* __restrict__ coo_col = mat.coo_col_indices.data();
    int coo_nnz = mat.coo_nnz;

    const float* __restrict__ x_ptr = x.data();
    float* __restrict__ y_ptr = y_out.data();

    double start = omp_get_wtime();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        
        // FASE 1: ELL (Inizializza y_ptr)
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int c = 0; c < max_nnz; c++) {
                int idx = r * max_nnz + c; // Row-Major obbligatorio per CPU
                int col = ell_col[idx];
                if (col != -1) {
                    sum += ell_val[idx] * x_ptr[col];
                }
            }
            y_ptr[r] = sum;
        }

        // FASE 2: COO (Overflow - Additivo)
        // Nota: COO può avere più entry per la stessa riga, quindi serve atomicità 
        // o riduzione. Atomic è più semplice qui anche se ha overhead.
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
    double avg_time_ms = ((end - start) / NUM_ITERATIONS) * 1000.0;

    std::cout << "OpenMP Hybrid Time: " << avg_time_ms << " ms" << std::endl;
}

void Smv_JDS_OpenMP(const JDSMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int rows = mat.rows;
    const float* values = mat.values.data();
    const int* col_indices = mat.col_indices.data();
    const int* jd_ptr = mat.jd_ptr.data();
    const int* row_lengths = mat.row_lengths.data();
    const int* row_perm = mat.row_perm.data();
    const float* x_ptr = x.data();
    float* y_ptr = y_out.data();

    double start = omp_get_wtime();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        
        // Parallelizziamo sulle righe della matrice ordinata (JDS)
        // Schedule dynamic potrebbe aiutare se le righe sono ancora molto diverse, 
        // ma JDS le ha già ordinate, quindi static va bene (le prime sono lunghe, le ultime corte).
        // Tuttavia, 'guided' è spesso ottimo per carichi decrescenti come JDS.
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            int len = row_lengths[i];
            
            for (int k = 0; k < len; k++) {
                // Accesso JDS: Salta alla diagonale k, offset i
                int idx = jd_ptr[k] + i;
                sum += values[idx] * x_ptr[col_indices[idx]];
            }
            
            // Scrittura risultato (Scatter)
            // Siccome 'row_perm[i]' è univoco per ogni 'i', nessun conflitto tra thread!
            y_ptr[row_perm[i]] = sum;
        }
    }

    double end = omp_get_wtime();
    double avg_time_ms = ((end - start) / NUM_ITERATIONS) * 1000.0;
    std::cout << "OpenMP JDS Time: " << avg_time_ms << " ms" << std::endl;
}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./omp_benchmark <matrix.mtx> <ell_cutoff>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int cutoff = std::stoi(argv[2]);
    int M, N, nnz;

    // Imposta numero thread OpenMP (opzionale, di solito auto-detect)
    // omp_set_num_threads(8); 
    std::cout << "OpenMP Threads Max: " << omp_get_max_threads() << std::endl;

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
    //ELLMatrix mat_ell = convert_to_ell(raw_elements, M, mat_csr.row_ptr, false); 
    HybridMatrix mat_hyb = convert_to_hybrid(raw_elements, M, N, cutoff, false);
    JDSMatrix mat_jds = convert_to_jds(raw_elements, M, N, mat_csr.row_ptr);
    print_matrix_info(mat_csr, ELLMatrix() , mat_hyb, mat_jds);
    Smv_CSR_OpenMP(mat_csr, x, y_out);


    // ------------------------------------------
    // TEST ELL (CPU Version - Row Major)
    // ------------------------------------------
    // ATTENZIONE: Passiamo 'false' per il layout, la CPU vuole Row-Major!
    
    //if (mat_ell.rows > 0) { // Se non ha fallito per troppa memoria
        
    //    Smv_ELL_OpenMP(mat_ell, x, y_out);
    //} else {
    //    std::cout << "ELL Skipped (Too large for memory)" << std::endl;
    //}

   

    // ------------------------------------------
    // TEST HYBRID (CPU Version - Row Major)
    // ------------------------------------------
    // Anche qui, passiamo 'false' per il layout ELL interno
    
    
    float coo_ratio = (float)mat_hyb.coo_nnz / (float)nnz * 100.0f;


    Smv_Hybrid_OpenMP(mat_hyb, x, y_out);
    Smv_JDS_OpenMP(mat_jds, x, y_out);
    return 0;
}