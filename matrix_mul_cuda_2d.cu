/*
 * Multiplying a 2D matrix using CUDA 
 */

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mul( int *a, int *b, int *c, int m, int n, int k){
  int row = blockIdx.y + blockDim.y * threadIdx.y;
  int col = blockIdx.x + blockDim.x * threadIdx.x;
  int sum = 0;

  if(col < k && row < m){
    for(int i = 0; i < n; i++){
      sum += a[row*n + i] * b[i*k + col];
    }
    c[row * k + col] = sum;
  }
}

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
  for (int i = 0; i < m; ++i) 
  {
    for (int j = 0; j < k; ++j) 
    {
      int tmp = 0.0;
      for (int h = 0; h < n; ++h) 
      {
        tmp += h_a[i * n + h] * h_b[h * k + j];
      }
      h_result[i * k + j] = tmp;
    }
  }
}

int main(){
  int m,n,k; // m=rows of 1st , n= cols of 1st and rows of 2nd , k = cols of 2nd
  srand(3333);

  m = 1024;
  n = 1024;
  k = 1024;

  //Allocate memory in host RAM, h_cc is used to store CPU results
  int *h_a, *h_b, *h_c , *h_cc;

  cudaMallocHost((void **) &h_a , sizeof(int)*m*n);
  cudaMallocHost((void **) &h_b , sizeof(int)*n*k);
  cudaMallocHost((void **) &h_c , sizeof(int)*m*k);
  cudaMallocHost((void **) &h_cc , sizeof(int)*m*k);

  //random initialie matrix A

  for(int i= 0; i < m; ++i){
    for(int j=0; j < n; ++j){
      h_a[i*n + j ] = rand() %1024;
    }
  }

  //Random intialize B
  for( int i = 0; i<n; ++i){
    for( int j = 0; j < n; j++){
      h_b[i*k + j] = rand()%1024;
    }
  }
  
  float gpu_elapsed_time_ms;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  
  //Allocate Memory on the device
  int *d_a, *d_b, *d_c;
  cudaMalloc((void **) &d_a, sizeof(int)*m*n);
  cudaMalloc((void **) &d_b, sizeof(int)*n*k);
  cudaMalloc((void **) &d_c, sizeof(int)*m*k);
 
  //Copy matrix A and B from host to device memory
  cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

  unsigned int grid_rows = (m + BLOCK_SIZE -1 ) / BLOCK_SIZE;
  unsigned int grid_cols = (k + BLOCK_SIZE -1 )/ BLOCK_SIZE;

  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE);

  gpu_matrix_mul<<<dimGrid, dimBlock>>>(
        d_a, d_b, d_c, m, n, k 
    );
  cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  
  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("Time Elapsed on matrix multiplication of %dx%d . %dx%d on GPU : %fms.\n\n", m,n,n,n,k, gpu_elapsed_time_ms);

  float cpu_elapsed_time_ms;

  cudaEventRecord(start, 0);

  cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
  printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);

}
