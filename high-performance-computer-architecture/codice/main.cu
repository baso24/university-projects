#include <iostream>
#include "mtx_reader.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include <iomanip>
#include <device_launch_parameters.h>

__global__ void spmv_csr_kernel(int num_rows, const int* row_ptr, const int* col_indices, const float* values, const float* x, float* y) {
    // assegno a giascun thread una riga
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // se il therad ha una riga assegnata e non è uscito => lavora
    if (row < num_rows) {
        float dot_product = 0.0f;
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
            dot_product += values[i] * x[col_indices[i]];
        }
        y[row] = dot_product;
    }
}
__global__ void spmv_ell_kernel(int num_rows, float *data, int *col_index,int num_elem, float *x, float *y) {
int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        for (int i = 0; i < num_elem; i++) {
            dot += data[row+i*num_rows] * x[col_index[row+i*num_rows]];
        }
    y[row] += dot;
    }
}
__global__ void spmv_coo_kernel(int nnz, const float*  values,const int* row_indices,const int* col_indices,const float* x,float* y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nnz) {
        int row = row_indices[idx];
        int col = col_indices[idx];
        float val = values[idx];

        // Moltiplicazione e accumulo atomico sul risultato parziale di ELL
        atomicAdd(&y[row], val * x[col]);
    }
}
__global__ void spmv_jds_kernel(int rows, 
                                const float* values, 
                                const int* col_indices, 
                                const int* jd_ptr,
                                const int* row_lengths,
                                const int* row_perm,
                                const float* x, 
                                float* y) 
{
    // 'row' qui è l'indice della riga nella matrice PERMUTATA (ordinata)
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;
        // len è il numero di elemnti nella riga
        int len = row_lengths[row]; 
        // itera lungo le "diagonali"
        for (int k = 0; k < len; k++) {
            // indice JDS = inizio_diag_k + offset_riga
            int idx = jd_ptr[k] + row;
            sum += values[idx] * x[col_indices[idx]];
        }
        
        // scrivo il risultato nella posizione originale usando il vettore permutazione
        y[row_perm[row]] = sum;
    }
}


float Smv_CSR_GPU( CSRMatrix &mat_csr,int nnz, const std::vector<float>& x, std::vector<float>& y_out){ 
    int *d_row_ptr, *d_col_indices;
    float *d_values, *d_x, *d_y;
    // alloco memoria sulla GPU dei dati della matrice e dei vettori
    cudaMalloc(&d_row_ptr, (mat_csr.rows + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_x, mat_csr.cols * sizeof(float));
    cudaMalloc(&d_y, mat_csr.rows * sizeof(float));
    //copio i dati sulla GPU
    cudaMemcpy(d_row_ptr, mat_csr.row_ptr.data(), (mat_csr.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, mat_csr.col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, mat_csr.values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), mat_csr.cols * sizeof(float), cudaMemcpyHostToDevice);

    // uso 256 thread per blocco e calcolo il numero di blocchi necessari
    int threads = 256;
    int blocks = (mat_csr.rows + threads - 1) / threads;
    // giro a vuoto per inizializzare la GPU 
    spmv_csr_kernel<<<blocks, threads>>>(mat_csr.rows, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    // aggiungo timer e inizio a contare
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    // eseguo 1000 iterazioni per avere un tempo piu stabile
    for (int i = 0 ; i < 1000; i++){
        spmv_csr_kernel<<<blocks, threads>>>(mat_csr.rows, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    // copio il risultato
    std::vector<float> h_y(mat_csr.rows);
    cudaMemcpy(h_y.data(), d_y, mat_csr.rows * sizeof(float), cudaMemcpyDeviceToHost);
    y_out = h_y;
    // pulisco la memoria
    cudaFree(d_row_ptr); cudaFree(d_col_indices); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms/1000.0f;

}

float Smv_ELL_GPU( const ELLMatrix& mat_ell, const std::vector<float>& x, std::vector<float>& y_out){
    int *d_col;
    float *d_values, *d_x, *d_y;

    // alloco memoria sulla GPU dei dati della matrice e dei vettori
    size_t matrix_size = mat_ell.values.size();

    cudaMalloc(&d_col, matrix_size * sizeof(int));
    cudaMalloc(&d_values, matrix_size * sizeof(float));
    cudaMalloc(&d_x, mat_ell.col_indices.size() * sizeof(float));
    cudaMalloc(&d_y, mat_ell.rows * sizeof(float));

    // copio i dati sulla GPU
    cudaMemcpy(d_col, mat_ell.col_indices.data(), matrix_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, mat_ell.values.data(), matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), mat_ell.cols * sizeof(float), cudaMemcpyHostToDevice);

    // uso 256 thread per blocco e calcolo il numero di blocchi necessari
    int threadsPerBlock = 256;
    int blocksPerGrid = (mat_ell.rows + threadsPerBlock - 1) / threadsPerBlock;
    // giro a vuoto per inizializzare la GPU 
    spmv_ell_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_ell.rows, d_values, d_col, mat_ell.max_nnz_per_row, d_x, d_y);
    // aggiungo timer e inizio a contare
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    // 1000 iterazioni 
    for (int i = 0 ; i < 1000; i++)
        spmv_ell_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_ell.rows, d_values, d_col, mat_ell.max_nnz_per_row, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    // copio il risultato
    std::vector<float> h_y(mat_ell.rows);
    cudaMemcpy(h_y.data(), d_y, mat_ell.rows * sizeof(float), cudaMemcpyDeviceToHost);
    y_out = h_y;
    // Pulizia
    cudaFree(d_col); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms/1000.0f;
}

float Smv_ELL_COO_GPU(const HybridMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int N = mat.rows;
    
    // setup ell
    int ell_width = mat.ell_part.max_nnz_per_row;
    size_t size_ell = mat.ell_part.values.size();

    float *d_ell_val, *d_x, *d_y;
    int *d_ell_col;
    // alloco memoria
    cudaMalloc(&d_ell_val, size_ell * sizeof(float));
    cudaMalloc(&d_ell_col, size_ell * sizeof(int));
    cudaMalloc(&d_x, mat.cols * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    // copio in GPU
    cudaMemcpy(d_ell_val, mat.ell_part.values.data(), size_ell * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell_col, mat.ell_part.col_indices.data(), size_ell * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

    // setup COO
    float *d_coo_val = nullptr;
    int *d_coo_row = nullptr;
    int *d_coo_col = nullptr;
    
    if (mat.coo_nnz > 0) {
        // alloco memoria per coo
        cudaMalloc(&d_coo_val, mat.coo_nnz * sizeof(float));
        cudaMalloc(&d_coo_row, mat.coo_nnz * sizeof(int));
        cudaMalloc(&d_coo_col, mat.coo_nnz * sizeof(int));
        // copio in GPU
        cudaMemcpy(d_coo_val, mat.coo_values.data(), mat.coo_nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_row, mat.coo_row_indices.data(), mat.coo_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coo_col, mat.coo_col_indices.data(), mat.coo_nnz * sizeof(int), cudaMemcpyHostToDevice);
    }

    
    int threadsPerBlock = 256;
    int blocksELL = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blocksCOO = (mat.coo_nnz + threadsPerBlock - 1) / threadsPerBlock;
    // giro a vuoto
    // ell
    spmv_ell_kernel<<<blocksELL, threadsPerBlock>>>(mat.rows, d_ell_val, d_ell_col, ell_width, d_x, d_y);
    // se presente coo
    if (mat.coo_nnz > 0) {
        spmv_coo_kernel<<<blocksCOO, threadsPerBlock>>>(mat.coo_nnz, d_coo_val, d_coo_row, d_coo_col, d_x, d_y);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < 1000; i++)
    {
        // ell
        spmv_ell_kernel<<<blocksELL, threadsPerBlock>>>(mat.rows, d_ell_val, d_ell_col, ell_width, d_x, d_y);
        // se presente coo
        if (mat.coo_nnz > 0) {
            spmv_coo_kernel<<<blocksCOO, threadsPerBlock>>>(mat.coo_nnz, d_coo_val, d_coo_row, d_coo_col, d_x, d_y);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(y_out.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // pulizia
    cudaFree(d_ell_val); cudaFree(d_ell_col); cudaFree(d_x); cudaFree(d_y);
    if (mat.coo_nnz > 0) {
        cudaFree(d_coo_val); cudaFree(d_coo_row); cudaFree(d_coo_col);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ms/1000.0f;
}

float Smv_JDS_GPU(const JDSMatrix& mat, const std::vector<float>& x, std::vector<float>& y_out) {
    int N = mat.rows;
    size_t nnz = mat.values.size();

    float *d_val, *d_x, *d_y;
    int *d_col, *d_jd_ptr, *d_row_perm, *d_row_len;
    // alloco memtodia
    cudaMalloc(&d_val, nnz * sizeof(float));
    cudaMalloc(&d_col, nnz * sizeof(int));
    cudaMalloc(&d_jd_ptr, (mat.max_nnz_per_row + 1) * sizeof(int));
    cudaMalloc(&d_row_perm, N * sizeof(int));
    cudaMalloc(&d_row_len, N * sizeof(int));
    cudaMalloc(&d_x, mat.cols * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    // copio nella GPU
    cudaMemcpy(d_val, mat.values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, mat.col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jd_ptr, mat.jd_ptr.data(), (mat.max_nnz_per_row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_perm, mat.row_perm.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_len, mat.row_lengths.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    // giro a vuoto
    spmv_jds_kernel<<<blocks, threads>>>(N, d_val, d_col, d_jd_ptr, d_row_len, d_row_perm, d_x, d_y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i=0; i<1000; i++) { 
        spmv_jds_kernel<<<blocks, threads>>>(N, d_val, d_col, d_jd_ptr, d_row_len, d_row_perm, d_x, d_y);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(y_out.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_val); cudaFree(d_col); cudaFree(d_jd_ptr); cudaFree(d_row_perm); cudaFree(d_row_len); cudaFree(d_x); cudaFree(d_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ms / 1000.0f; // Media
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./benchmark <matrix.mtx>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int cutoff = std::stoi(argv[2]);
    int M, N, nnz;

    // lettura del file
    std::cout << "Reading " << filename << "..." << std::endl;
    auto raw_elements = read_mtx_file(filename, M, N, nnz);

    // creao CSR
    CSRMatrix mat_csr = convert_to_csr(raw_elements, M, N);
    // creo ELL (column-major per GPU)
    ELLMatrix mat_ell = convert_to_ell(raw_elements, M, mat_csr.row_ptr, true);
    // creo HYBRID
    HybridMatrix mat_hyb = convert_to_hybrid(raw_elements, M, N, cutoff, true);
    // creo JDS
    JDSMatrix mat_jds = convert_to_jds(raw_elements, M, N, mat_csr.row_ptr);
    // stampo info della dimensione
    print_matrix_info(mat_csr, mat_ell, mat_hyb, mat_jds);
    // creo vettore di input x (dimensione N) valore 1 perché non è importante il risultato
    std::vector<float> x(N, 1.0f); 
    // creo vettori di output per ogni formato
    std::vector<float> y_csr(M, 0.0f);
    std::vector<float> y_ell(M, 0.0f);
    std::vector<float> y_hyb(M, 0.0f);
    std::vector<float> y_jds(M, 0.0f);
    // CSR
    float time_csr = Smv_CSR_GPU(mat_csr,nnz,x, y_csr);
    // ELL
    float time_ell = Smv_ELL_GPU(mat_ell, x, y_ell);
    // HYBRID
    float time_hyb = Smv_ELL_COO_GPU(mat_hyb, x, y_hyb);
    // JDS
    float time_jds = Smv_JDS_GPU(mat_jds, x, y_jds);

    // stampa risultati
    std::cout << "CUDA CSR Time: " << time_csr << " ms" << std::endl;
    std::cout << "CUDA ELL Time: " << time_ell << " ms" << std::endl;
    std::cout << "CUDA Hybrid Time: " << time_hyb << " ms" << std::endl;
    std::cout << "CUDA JDS Time: " << time_jds << " ms" << std::endl;

    return 0;
}