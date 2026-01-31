#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

// Formato Compressed Sparse Row (CSR)
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int num_nonzeros;
    vector<float> data;      // Valori non-zero
    vector<int> col_index;   // Indici di colonna
    vector<int> row_ptr;     // Puntatori all'inizio di ogni riga
};

// Formato ELLPACK (ELL)
struct ELLMatrix {
    int num_rows;
    int num_elem;                 // Max non-zero per riga (K)
    vector<float> data;      // Paddata e Trasposta (Column-Major)
    vector<int> col_index;
};

// Formato Coordinate (COO)
struct COOMatrix {
    int num_nonzeros;
    vector<float> data;
    vector<int> col_index;
    vector<int> row_index;
};

// Formato Jagged Diagonal Storage (JDS)
struct JDSMatrix {
    int num_rows;
    vector<float> data;
    vector<int> col_index;
    vector<int> jds_row_index;   // Mappa riga ordinata -> riga originale
    vector<int> jds_section_ptr; // Delimita sezioni con stessa lunghezza
};

// Generazione matrice sparsa casuale in formato CSR
CSRMatrix generate_big_matrix(int n, int k) {
    CSRMatrix csr;
    csr.num_rows = n;
    csr.num_cols = n;
    csr.row_ptr.push_back(0);

    for (int i = 0; i < n; ++i) {
        // Generiamo k elementi attorno alla diagonale
        for (int j = max(0, i - k / 2); j < min(n, i + k / 2 + 1); ++j) {
            csr.data.push_back(static_cast<float>(rand()) / RAND_MAX);
            csr.col_index.push_back(j);
        }
        csr.row_ptr.push_back(csr.data.size());
    }
    csr.num_nonzeros = csr.data.size();
    return csr;
}

// Conversione CSR -> ELL con Padding e Transposizione
ELLMatrix convert_csr_to_ell(const CSRMatrix& csr) {
    ELLMatrix ell;
    ell.num_rows = csr.num_rows;
    
    // Trova la riga più lunga (K)
    int max_nz = 0;
    for (int i = 0; i < csr.num_rows; ++i) {
        max_nz = max(max_nz, csr.row_ptr[i+1] - csr.row_ptr[i]);
    }
    ell.num_elem = max_nz;

    // Allocazione Column-Major: [num_rows * num_elem]
    ell.data.assign(ell.num_rows * ell.num_elem, 0.0f);
    ell.col_index.assign(ell.num_rows * ell.num_elem, -1);

    for (int r = 0; r < csr.num_rows; ++r) {
        int row_start = csr.row_ptr[r];
        int row_end = csr.row_ptr[r+1];
        for (int i = 0; i < (row_end - row_start); ++i) {
            // Indice Column-Major per favorire la coalescenza in CUDA
            int target_idx = r + i * ell.num_rows;
            ell.data[target_idx] = csr.data[row_start + i];
            ell.col_index[target_idx] = csr.col_index[row_start + i];
        }
    }
    return ell;
}

// Conversione CSR -> JDS (Sorting delle righe per lunghezza)
JDSMatrix convert_csr_to_jds(const CSRMatrix& csr) {
    JDSMatrix jds;
    jds.num_rows = csr.num_rows;
    
    // Creiamo un vettore di coppie (lunghezza riga, indice originale)
    vector<pair<int, int>> row_lengths(csr.num_rows);
    for (int i = 0; i < csr.num_rows; ++i) {
        row_lengths[i] = {csr.row_ptr[i+1] - csr.row_ptr[i], i};
    }

    // Ordiniamo le righe dalla più lunga alla più corta
    sort(row_lengths.rbegin(), row_lengths.rend());

    jds.jds_row_index.resize(csr.num_rows);
    jds.data.reserve(csr.num_nonzeros);
    jds.col_index.reserve(csr.num_nonzeros);
    
    // Ricostruiamo la matrice seguendo il nuovo ordine
    for (int i = 0; i < csr.num_rows; ++i) {
        int original_row = row_lengths[i].second;
        jds.jds_row_index[i] = original_row;
        
        for (int j = csr.row_ptr[original_row]; j < csr.row_ptr[original_row+1]; ++j) {
            jds.data.push_back(csr.data[j]);
            jds.col_index.push_back(csr.col_index[j]);
        }
    }
    return jds;
}

// Conversione CSR -> COO
COOMatrix convert_csr_to_coo(const CSRMatrix& csr) {
    COOMatrix coo;
    coo.num_nonzeros = csr.num_nonzeros;
    coo.data = csr.data;
    coo.col_index = csr.col_index;
    coo.row_index.reserve(csr.num_nonzeros);
    for (int i = 0; i < csr.num_rows; ++i) {
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i+1]; ++j) {
            coo.row_index.push_back(i);
        }
    }
    return coo;
}

// SpMV CSR sequenziale
void spmv_csr_seq(const CSRMatrix& A, const vector<float>& x, vector<float>& y) {
    for (int row = 0; row < A.num_rows; ++row) {
        float dot = 0;
        for (int elem = A.row_ptr[row]; elem < A.row_ptr[row+1]; ++elem) {
            dot += A.data[elem] * x[A.col_index[elem]];
        }
        y[row] += dot;
    }
}

// SpMV ELL sequenziale
void spmv_ell_seq(const ELLMatrix& A, const vector<float>& x, vector<float>& y) {
    for (int r = 0; r < A.num_rows; ++r) {
        float dot = 0;
        for (int i = 0; i < A.num_elem; ++i) {
            int idx = r + i * A.num_rows;
            if (A.col_index[idx] != -1) {
                dot += A.data[idx] * x[A.col_index[idx]];
            }
        }
        y[r] += dot;
    }
}

// SpMV COO sequenziale
void spmv_coo_seq(const COOMatrix& A, const vector<float>& x, vector<float>& y) {
    for (int i = 0; i < A.num_nonzeros; ++i) {
        y[A.row_index[i]] += A.data[i] * x[A.col_index[i]];
    }
}

// SpMV CSR OpenMP
void spmv_csr_omp(const CSRMatrix& A, const vector<float>& x, vector<float>& y) {
    #pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < A.num_rows; ++row) {
        float dot = 0;
        for (int elem = A.row_ptr[row]; elem < A.row_ptr[row+1]; ++elem) {
            dot += A.data[elem] * x[A.col_index[elem]];
        }
        y[row] += dot;
    }
}

// SpMV ELL OpenMP
void spmv_ell_omp(const ELLMatrix& A, const vector<float>& x, vector<float>& y) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < A.num_rows; ++r) {
        float dot = 0;
        for (int i = 0; i < A.num_elem; ++i) {
            int idx = r + i * A.num_rows;
            if (A.col_index[idx] != -1) {
                dot += A.data[idx] * x[A.col_index[idx]];
            }
        }
        y[r] += dot;
    }
}

// SpMV COO OpenMP
void spmv_coo_omp(const COOMatrix& A, const vector<float>& x, vector<float>& y) {
    #pragma omp parallel for
    for (int i = 0; i < A.num_nonzeros; ++i) {
        int row = A.row_index[i];
        int col = A.col_index[i];
        float val = A.data[i];
        #pragma omp atomic
        y[row] += val * x[col];
    }
}

// Funzione di verifica con tolleranza per float
bool verify_result(const vector<float>& ref, const vector<float>& res) {
    if (ref.size() != res.size()) return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (abs(ref[i] - res[i]) > 1e-4) return false;
    }
    return true;
}

int main() {
    int N = 1000000; // 100k righe
    int K = 32;     // 32 elementi non-zero per riga
    
    CSRMatrix m = generate_big_matrix(N, K);
    vector<float> x(N, 1.0f);
    vector<float> y_ref(N, 0.0f);

    cout << "--- Fase 1: Calcoli e trasformazioni ---" << endl;
    spmv_csr_seq(m, x, y_ref);

    ELLMatrix m_ell = convert_csr_to_ell(m);
    vector<float> y_ell(N, 0.0f);
    spmv_ell_seq(m_ell, x, y_ell);
    
    cout << "Verifica ELL: " << (verify_result(y_ref, y_ell) ? "OK" : "ERRORE") << endl;

    COOMatrix m_coo = convert_csr_to_coo(m);
    vector<float> y_coo(N, 0.0f);
    spmv_coo_seq(m_coo, x, y_coo);

    cout << "Verifica COO: " << (verify_result(y_ref, y_coo) ? "OK" : "ERRORE") << endl;

    // Verifica OpenMP
    vector<float> y_omp(N, 0.0f);
    spmv_csr_omp(m, x, y_omp);
    cout << "Verifica CSR OpenMP: " << (verify_result(y_ref, y_omp) ? "OK" : "ERRORE") << endl;

    fill(y_omp.begin(), y_omp.end(), 0.0f);
    spmv_ell_omp(m_ell, x, y_omp);
    cout << "Verifica ELL OpenMP: " << (verify_result(y_ref, y_omp) ? "OK" : "ERRORE") << endl;

    fill(y_omp.begin(), y_omp.end(), 0.0f);
    spmv_coo_omp(m_coo, x, y_omp);
    cout << "Verifica COO OpenMP: " << (verify_result(y_ref, y_omp) ? "OK" : "ERRORE") << endl;

    cout << "\n--- Fase 2: Benchmark Prestazioni (100 iterazioni) ---" << endl;
    int iterations = 100;

    // Benchmark CSR
    fill(y_ref.begin(), y_ref.end(), 0.0f); // Reset
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) spmv_csr_seq(m, x, y_ref);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Tempo CSR Sequenziale: " << elapsed.count() << " s" << endl;

    // Benchmark CSR OpenMP
    fill(y_ref.begin(), y_ref.end(), 0.0f);
    start = chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) spmv_csr_omp(m, x, y_ref);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Tempo CSR OpenMP:    " << elapsed.count() << " s" << endl;

    // Benchmark ELL
    fill(y_ell.begin(), y_ell.end(), 0.0f); // Reset
    start = chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) spmv_ell_seq(m_ell, x, y_ell);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Tempo ELL Sequenziale: " << elapsed.count() << " s" << endl;

    // Benchmark ELL OpenMP
    fill(y_ell.begin(), y_ell.end(), 0.0f);
    start = chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) spmv_ell_omp(m_ell, x, y_ell);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Tempo ELL OpenMP:    " << elapsed.count() << " s" << endl;

    // Benchmark COO
    fill(y_coo.begin(), y_coo.end(), 0.0f); // Reset
    start = chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) spmv_coo_seq(m_coo, x, y_coo);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Tempo COO Sequenziale: " << elapsed.count() << " s" << endl;

    // Benchmark COO OpenMP
    fill(y_coo.begin(), y_coo.end(), 0.0f);
    start = chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) spmv_coo_omp(m_coo, x, y_coo);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Tempo COO OpenMP:    " << elapsed.count() << " s" << endl;

    cout << "\n--- Fase 3: Trasformazione JDS (Sorting) ---" << endl;
    JDSMatrix m_jds = convert_csr_to_jds(m);
    // Nota: JDS richiede un passaggio extra per riordinare y finale se usato in solver iterativi
    cout << "Matrice JDS preparata (sorting righe completato)." << endl;

    return 0;
}