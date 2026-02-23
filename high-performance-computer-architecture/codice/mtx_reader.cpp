#include "mtx_reader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

// compare per il sort
bool compareElements(const COOElement& a, const COOElement& b) {
    if (a.row != b.row) 
        return a.row < b.row;
    return a.col < b.col;
}

std::vector<COOElement> read_mtx_file(const std::string& filename, int& M, int& N, int& nnz) {
    //apre il file
    std::ifstream file(filename);
    // controlla se è aperto correttamente
    if (!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    // nell'header dei file .mtx c'è la parolaa symmetric che indica che la matrice è simmetrica
    // es. in cant.mtx
    // %%MatrixMarket matrix coordinate real symmetric
    bool is_symmetric = false;

    // controlla se è simmetrica => se si devo specchiare gli elementi non nella diagonale
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            if (line.find("symmetric") != std::string::npos) 
                // se trovo la parola => simmetrico
                is_symmetric = true;
            // leggo solo se è presente symmetric, il resto dell'header lo skippo
            continue;
        }
        // appena leggo la prima riga non commentata con % => header del file con M(righe), N(colonne), nnz(se simmetrico è la metà)
        std::stringstream ss(line);
        ss >> M >> N >> nnz;
        break;
    }
    //creo vettore di COO 
    std::vector<COOElement> elements;
    // Riserva memoria stimata (se simmetrica sarà il doppio alla fine, ma è un'ottimizzazione)
    elements.reserve(nnz);

    int r, c;
    float v;
    // Lettura dati
    while (file >> r >> c >> v) {
        // il file è indicizzato a 1 => sottraggo 1 per avere primo indice 0
        r--; c--; 
        elements.push_back({r, c, v});
        // se è simmetrica devo aggiungere anche l'elemento specchiato se non è nella diagonale
        // se r == c => diagonale
        if (is_symmetric && r != c) {
            elements.push_back({c, r, v});
        }
    }
    
    // nnz è il numero di elementi salvati nel vettore
    nnz = elements.size();

    // ordina Row-Major
    std::sort(elements.begin(), elements.end(), compareElements);

    return elements;
}

CSRMatrix convert_to_csr(const std::vector<COOElement>& elements, int M, int N) {
    CSRMatrix csr;
    csr.rows = M;
    csr.cols = N;
    csr.nnz = elements.size();
    
    csr.values.reserve(csr.nnz);
    csr.col_indices.reserve(csr.nnz);
    csr.row_ptr.assign(M + 1, 0);
    // conversione semplice
    // aggiungo valore, colonna e incremento il contatore degli elementi di quella colonna
    // a questo stage row_ptr[i] contene il numero di elementi della riga i
    for (const auto& el : elements) {
        csr.values.push_back(el.val);
        csr.col_indices.push_back(el.col);
        csr.row_ptr[el.row + 1]++;
    }

    // per far si che row_ptr abbia l'indice di inizio di raga
    // aggiungo ad ogni elemento il valore di quello precedente partendo dal primo
    for (int i = 0; i < M; i++) {
        // 
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    return csr;
}

ELLMatrix convert_to_ell(const std::vector<COOElement>& elements, int M, const std::vector<int>& row_ptr, bool use_gpu_layout) {
    ELLMatrix ell;
    ell.rows = M;
    ell.is_column_major = use_gpu_layout;

    // calcolo il numero massimo di elementi per riga (mi serve per definire la larghezza)
    // baro un po' usando il vettore calcolato per CSR
    int max_nnz = 0;
    for (int i = 0; i < M; i++) {
        int row_len = row_ptr[i+1] - row_ptr[i];
        if (row_len > max_nnz) 
            max_nnz = row_len;
    }
    ell.max_nnz_per_row = max_nnz;

    // alloco spazio per la matrice
    int total_elements = M * max_nnz; 
    ell.values.assign(total_elements, 0.0f);
    ell.col_indices.assign(total_elements, -1); // -1 indica padding

    // riempio la matrice
    // per ogni riga, tengo conto del numero di elementi inseriti
    std::vector<int> current_col_in_row(M, 0);
    
    for (const auto& el : elements) {
        int r = el.row;
        int c_idx = current_col_in_row[r]; 

        int flat_index;

        if (use_gpu_layout) {
            // COLUMN-MAJOR (Per CUDA: coalescing)
            // l'elemento (riga, col_ell) si trova a riga + col_ell * num_righe
            flat_index = r + (c_idx * M);
        } else {
            // ROW-MAJOR (Per CPU: cache locality)
            // l'elemento si trova a riga * max_nnz + col_ell
            flat_index = (r * max_nnz) + c_idx;
        }
        
        ell.values[flat_index] = el.val;
        ell.col_indices[flat_index] = el.col;
        
        current_col_in_row[r]++;
    }

    return ell;
}
HybridMatrix convert_to_hybrid(const std::vector<COOElement>& elements, int M, int N, int max_ell_width, bool use_gpu_layout) {
    HybridMatrix hyb;
    hyb.rows = M;
    hyb.cols = N;
    
    // parte ell
    hyb.ell_part.rows = M;
    hyb.ell_part.cols = N;
    hyb.ell_part.max_nnz_per_row = max_ell_width;
    hyb.ell_part.is_column_major = use_gpu_layout;
    
    // alloca memoria ell
    size_t ell_size = (size_t)M * max_ell_width;
    hyb.ell_part.values.assign(ell_size, 0.0f);
    hyb.ell_part.col_indices.assign(ell_size, -1); // -1 = padding

    // alloca memoria COO (stima)
    // Stimiamo una dimensione iniziale per evitare troppe riallocazioni
    hyb.coo_values.reserve(elements.size() / 10); 
    hyb.coo_row_indices.reserve(elements.size() / 10);
    hyb.coo_col_indices.reserve(elements.size() / 10);

    // riempio matici ell e coo
    // per ell tengo conto di qeuanti elementi ho inserito per ogni riga
    std::vector<int> current_col_in_row(M, 0);

    for (const auto& el : elements) {
        int r = el.row;
        int current_pos = current_col_in_row[r];
        // se l'elemento a riga r supera il limite => aggiungo in coo
        if (current_pos < max_ell_width) {
            // minore => inserisco in ell
            // codice come sopra
            size_t idx;
            if (use_gpu_layout) {
                // Column-Major (r + c * rows)
                idx = (size_t)r + ((size_t)current_pos * (size_t)M);
            } else {
                // Row-Major
                idx = ((size_t)r * (size_t)max_ell_width) + (size_t)current_pos;
            }
            hyb.ell_part.values[idx] = el.val;
            hyb.ell_part.col_indices[idx] = el.col;
        } else {
            // aggingo in coo
            // aggiungo e basta perché gli elementi sono già orfinati in coo
            hyb.coo_row_indices.push_back(r);
            hyb.coo_col_indices.push_back(el.col);
            hyb.coo_values.push_back(el.val);
        }
        
        current_col_in_row[r]++;
    }

    hyb.coo_nnz = hyb.coo_values.size();
    return hyb;
}
JDSMatrix convert_to_jds(const std::vector<COOElement>& elements, int M, int N, const std::vector<int>& row_ptr_csr) {
    JDSMatrix jds;
    jds.rows = M;
    jds.cols = N;

    // calcola lunghezze e crea coppie (lunghezza, indice_originale)
    std::vector<std::pair<int, int>> row_info(M);
    int max_nnz = 0;
    for (int i = 0; i < M; i++) {
        int len = row_ptr_csr[i+1] - row_ptr_csr[i];
        row_info[i] = {len, i};
        if (len > max_nnz) max_nnz = len;
    }
    jds.max_nnz_per_row = max_nnz;

    // orfina le righe in base alla lunghezza (decrescente)
    std::sort(row_info.begin(), row_info.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first > b.first; // decrescente
    });

    // riempi row_perm e row_lengths
    jds.row_perm.resize(M);
    jds.row_lengths.resize(M);
    std::vector<int> original_to_permuted(M); // Mappa inversa temporanea
    
    for (int i = 0; i < M; i++) {
        jds.row_lengths[i] = row_info[i].first;
        jds.row_perm[i] = row_info[i].second;
        original_to_permuted[row_info[i].second] = i;
    }

    // alloco spazio per jds
    jds.values.reserve(elements.size());
    jds.col_indices.reserve(elements.size());
    jds.jd_ptr.assign(max_nnz + 1, 0);

    // Dobbiamo riempire i dati "per colonne" della matrice permutata.
    // Usiamo vettori temporanei per ogni riga permutata per facilitare l'accesso
    std::vector<std::vector<COOElement>> rows_data(M);
    for (const auto& el : elements) {
        int perm_row = original_to_permuted[el.row];
        rows_data[perm_row].push_back(el);
    }
    
    // Ordina col_indices dentro ogni riga (opzionale ma consigliato)
    for(auto& r : rows_data) {
        std::sort(r.begin(), r.end(), [](const COOElement& a, const COOElement& b){ return a.col < b.col; });
    }

    // Riempimento JDS: Loop esterno sulle "diagonali" (k), interno sulle righe (i)
    int current_pos = 0;
    for (int k = 0; k < max_nnz; k++) {
        jds.jd_ptr[k] = current_pos;
        
        // Itera su tutte le righe che hanno almeno k+1 elementi
        // Grazie al sort, sono le prime N righe finché length > k
        for (int i = 0; i < M; i++) {
            if (k < jds.row_lengths[i]) {
                // Prendi il k-esimo elemento della riga i-esima (permutata)
                jds.values.push_back(rows_data[i][k].val);
                jds.col_indices.push_back(rows_data[i][k].col);
                current_pos++;
            } else {
                // Visto che sono ordinate, se questa riga è finita, sono finite tutte le successive
                break; 
            }
        }
    }
    jds.jd_ptr[max_nnz] = current_pos; // End pointer

    return jds;
}
void print_matrix_info(const CSRMatrix& csr, const ELLMatrix& ell, const HybridMatrix& hyb, const JDSMatrix& jds) {
    // calcolo memoria allocata in MB
    size_t mem_csr = (csr.values.size() * sizeof(float)) + 
                     (csr.col_indices.size() * sizeof(int)) + 
                     (csr.row_ptr.size() * sizeof(int));
    
    size_t mem_ell = (ell.values.size() * sizeof(float)) + 
                     (ell.col_indices.size() * sizeof(int));
    size_t mem_hyb = (hyb.ell_part.values.size() * sizeof(float)) + 
                     (hyb.ell_part.col_indices.size() * sizeof(int)) +
                     (hyb.coo_values.size() * sizeof(float)) +
                     (hyb.coo_row_indices.size() * sizeof(int)) +
                     (hyb.coo_col_indices.size() * sizeof(int));
    size_t mem_jds = (jds.values.size() * sizeof(float)) +
                        (jds.col_indices.size() * sizeof(int)) +
                        (jds.jd_ptr.size() * sizeof(int)) +
                        (jds.row_perm.size() * sizeof(int)) +
                        (jds.row_lengths.size() * sizeof(int));

    std::cout << "--- Statistiche Matrice ---" << std::endl;
    std::cout << "Rows: " << csr.rows << ", NNZ: " << csr.nnz << std::endl;
    std::cout << "CSR Mem: " << mem_csr / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "ELL Mem: " << mem_ell / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "Hybrid Mem: " << mem_hyb / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "JDS Mem: " << mem_jds / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "ELL Layout: " << (ell.is_column_major ? "Column-Major (GPU)" : "Row-Major (CPU)") << std::endl;
    std::cout << "---------------------------" << std::endl;
}
