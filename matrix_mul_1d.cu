#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<assert.h>


using namespace::std;

__global__ 
void g_mat_mul( int *a, int *b, int *c, int m, int n, int k){
    //Kernel for matrix multiplication on GPU

  int index = threadIdx.x;
  for(int i=0; i < m; i++){
    //printf("\nValue of a is %d at index %d", a[(index/m)*m + i], (index/m)*m + i);
    //printf("\nValue of b is %d at index %d", b[(index%3) + i], (index%m) + i);
    c[index] += a[(index/m)*m + i] * b[(index % m) + i*m];
  }
  printf("\nValue for C at thread %d is %d\n", index, c[index]);
}

void cpu_matrix_mul(int *a, int *b, int *c, int m, int n, int k){
  for( int i = 0; i < m; i++){
    for( int j=0; j < k; j++){
      for( int l = 0; l < n; l++){
        c[i + m*j] += a[i + n*l]*b[n*l + j];
      }
    }
  }
}


int main(void){
  int m = 100, n = 100, k = 100;
  int size_a, size_b, size_c;


  size_a = m*n;
  size_b = n*k;
  size_c = m*k;

  //Allocated Memory on device for matrices
  // a = (m*n) , b = (n*k), c = (m*k)
  int *host_a, *host_b, *host_c; //cudaOperation
  int *a, *b, *c;

  a = (int *) malloc(sizeof(int)*size_a);
  b = (int *) malloc(sizeof(int)*size_b);
  c = (int *) malloc(sizeof(int)*size_c);

  cudaMallocManaged(&host_a, sizeof(int)*size_a);
  cudaMallocManaged(&host_b , sizeof(int)*size_b);
  cudaMallocManaged(&host_c , sizeof(int)*size_c);

  //Randomly initialize Matrices

  //Randomly initialize A
  std::cout<<"Initialized A";
  for(int i=0; i < m*n; i++){
    if(i%(m+1) == 0){
      host_a[i] =1 ;
      a[i] = 1;
    }else{
      host_a[i] = 0;
      a[i] = 0;
    }
  }
  std::cout<<"Initialized B";
  for(int i=0; i < n*k; i++){
    host_b[i] = i+1;
    b[i] = i + 1;
  }
    
  float gpu_elapsed_time, cpu_elapsed_time;
  cudaEvent_t start, stop;  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start,0);

  g_mat_mul<<<1,size_c>>>(
        host_a, host_b, host_c, m, n, k
  );
  //cpu_matrix_mul(a,b,c,m,n,k);
  cudaDeviceSynchronize();

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  
  cudaEventElapsedTime(&gpu_elapsed_time, start, stop);
  printf("\nTime elapsed on matrix multiplication on GPU: %f ms.\n\n", gpu_elapsed_time);
  cudaFree(host_a);
  cudaFree(host_b);
  cudaFree(host_c);

  cudaEventRecord(start,0);
  cpu_matrix_mul(a,b,c,m,n,k);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_elapsed_time, start, stop);
  printf("\nTime elapsed on matrix multiplication on CPU: %f ms.\n\n", cpu_elapsed_time);
}
