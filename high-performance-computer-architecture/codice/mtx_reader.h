#ifndef MTX_READER_H
#define MTX_READER_H

#include <vector>
#include <string>
#include <iostream>

// Struttura intermedia per leggere i dati grezzi
struct COOElement {
    int row, col;
    float val;
};

// Formato Compressed Sparse Row
struct CSRMatrix {
    int rows;
    int cols;
    int nnz;
    std::vector<float> values;      // array A
    std::vector<int> col_indices;   // array JA
    std::vector<int> row_ptr;       // array IA
};

// Formato ELLPACK
struct ELLMatrix {
    int rows;
    int cols;
    int max_nnz_per_row; // La "larghezza" della matrice ELL
    bool is_column_major; // TRUE per CUDA, FALSE per CPU

    // Nota: la dimensione di questi vettori sarà (rows * max_nnz_per_row)
    std::vector<float> values;      
    std::vector<int> col_indices;   
};
struct HybridMatrix {
    int rows;
    int cols;
    
    // Parte 1: ELL
    ELLMatrix ell_part; 
    
    // Parte 2: COO (Overflow)
    int coo_nnz;
    std::vector<float> coo_values;
    std::vector<int> coo_row_indices;
    std::vector<int> coo_col_indices;
};

struct JDSMatrix {
    int rows;
    int cols;
    int max_nnz_per_row;
    
    std::vector<float> values;      // Dati (ordinati per diagonali)
    std::vector<int> col_indices;   // Indici colonna
    std::vector<int> jd_ptr;        // Puntatori all'inizio di ogni diagonale (size = max_nnz + 1)
    std::vector<int> row_perm;      // Permutazione: mappa riga_JDS -> riga_Originale
    std::vector<int> row_lengths;   // Lunghezza di ogni riga permutata (utile per il kernel)
};

// Funzione di conversione
// --- Funzioni Esposte ---

// 1. Legge il file .mtx e restituisce un vettore di elementi COO grezzi
std::vector<COOElement> read_mtx_file(const std::string& filename, int& M, int& N, int& nnz);

// 2. Converte i dati COO in CSR
CSRMatrix convert_to_csr(const std::vector<COOElement>& elements, int M, int N);

// 3. Converte i dati COO in ELL
// Usa 'use_gpu_layout = true' quando prepari i dati per CUDA!
ELLMatrix convert_to_ell(const std::vector<COOElement>& elements, int M, const std::vector<int>& row_ptr, bool use_gpu_layout = false);

HybridMatrix convert_to_hybrid(const std::vector<COOElement>& elements, int M, int N, int max_ell_width, bool use_gpu_layout = true);

JDSMatrix convert_to_jds(const std::vector<COOElement>& elements, int M, int N, const std::vector<int>& row_ptr_csr);

// 4. Utility per stampare info sulle matrici
void print_matrix_info(const CSRMatrix& csr, const ELLMatrix& ell, const HybridMatrix& hyb, const JDSMatrix& jds);
// Funzione per calcolare un K intelligente (es. media NNZ * 2)
int calculate_hybrid_cutoff(int M, int nnz);

// Funzione di conversione

#endif // MTX_READER_H